#!/bin/bash
# Ray Autoscaler Setup Script for Metagenomic Clustering
# This script sets up everything needed for the Ray autoscaler environment

set -e

# Configuration
PROJECT_NAME="ray-metagenomics"
AWS_REGION="${AWS_REGION:-us-east-1}"
KEY_PAIR_NAME="my-tfm-aws-key"
SECURITY_GROUP_NAME="ray-metagenomics-sg"
IAM_ROLE_NAME="RayMetagenomicsRole"
IAM_PROFILE_NAME="RayMetagenomicsProfile"
S3_BUCKET_PREFIX="metagenomics"
FSX_SIZE="1200"  # GB

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running on Linux/macOS
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        log_error "This script requires Linux or macOS. Windows users should use WSL2."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    log_info "Python version: $PYTHON_VERSION"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip
    pip3 install --upgrade pip
    
    # Install Ray with full dependencies
    pip3 install "ray[default,serve,tune,data]==2.48.0"
    
    # Install AWS dependencies
    pip3 install boto3 awscli
    
    # Install other dependencies
    pip3 install biopython pandas numpy pyarrow pyyaml
    
    log_success "Python dependencies installed"
}

# Configure AWS CLI
configure_aws() {
    log_info "Configuring AWS CLI..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Installing..."
        
        # Install AWS CLI v2
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
            sudo installer -pkg AWSCLIV2.pkg -target /
            rm AWSCLIV2.pkg
        else
            # Linux
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip -q awscliv2.zip
            sudo ./aws/install
            rm -rf aws awscliv2.zip
        fi
    fi
    
    # Check if AWS is configured
    if ! aws sts get-caller-identity &>/dev/null; then
        log_warning "AWS credentials not configured"
        echo "Please run 'aws configure' to set up your AWS credentials"
        echo "You'll need:"
        echo "  - AWS Access Key ID"
        echo "  - AWS Secret Access Key"
        echo "  - Default region (suggest: $AWS_REGION)"
        echo "  - Default output format (suggest: json)"
        
        read -p "Press Enter after configuring AWS credentials..."
        
        # Verify configuration
        if ! aws sts get-caller-identity &>/dev/null; then
            log_error "AWS credentials still not working. Please check your configuration."
            exit 1
        fi
    fi
    
    # Set default region
    aws configure set default.region $AWS_REGION
    
    # Get account info
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    
    log_success "AWS configured - Account: $ACCOUNT_ID, User: $USER_ARN"
}

# Create AWS resources
create_aws_resources() {
    log_info "Creating AWS resources..."
    
    # Create key pair if it doesn't exist
    if ! aws ec2 describe-key-pairs --key-names $KEY_PAIR_NAME &>/dev/null; then
        log_info "Creating EC2 key pair: $KEY_PAIR_NAME"
        aws ec2 create-key-pair --key-name $KEY_PAIR_NAME --query 'KeyMaterial' --output text > ~/.ssh/$KEY_PAIR_NAME.pem
        chmod 600 ~/.ssh/$KEY_PAIR_NAME.pem
        log_success "Key pair created: ~/.ssh/$KEY_PAIR_NAME.pem"
    else
        log_info "Key pair already exists: $KEY_PAIR_NAME"
    fi
    
    # Get default VPC
    #VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query 'Vpcs[0].VpcId' --output text)
    VPC_ID=vpc-0c7504c28330cf6c5
    if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
        log_error "No default VPC found. Please create a VPC first."
        exit 1
    fi
    log_info "Using VPC: $VPC_ID"
    
    # Get default subnet
    #SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)
    SUBNET_ID=subnet-028667cdb554810f9
    if [ "$SUBNET_ID" == "None" ] || [ -z "$SUBNET_ID" ]; then
        log_error "No default subnet found. Please create a subnet first."
        exit 1
    fi
    log_info "Using subnet: $SUBNET_ID"
    
    # Create security group if it doesn't exist
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
    
    if [ "$SECURITY_GROUP_ID" == "None" ] || [ -z "$SECURITY_GROUP_ID" ]; then
        log_info "Creating security group: $SECURITY_GROUP_NAME"
        SECURITY_GROUP_ID=$(aws ec2 create-security-group \
            --group-name $SECURITY_GROUP_NAME \
            --description "Security group for Ray metagenomic cluster" \
            --vpc-id $VPC_ID \
            --query 'GroupId' --output text)
        
        # Add rules
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0  # SSH
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 6379 --source-group $SECURITY_GROUP_ID  # Ray
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8265 --cidr 0.0.0.0/0  # Dashboard
        aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 10001-10999 --source-group $SECURITY_GROUP_ID  # Ray workers
        
        log_success "Security group created: $SECURITY_GROUP_ID"
    else
        log_info "Security group already exists: $SECURITY_GROUP_ID"
    fi
    
    # Create IAM role if it doesn't exist
    if ! aws iam get-role --role-name $IAM_ROLE_NAME &>/dev/null; then
        log_info "Creating IAM role: $IAM_ROLE_NAME"
        
        # Trust policy
        cat > /tmp/trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
        
        aws iam create-role --role-name $IAM_ROLE_NAME --assume-role-policy-document file:///tmp/trust-policy.json
        
        # Permissions policy
        cat > /tmp/permissions-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::$S3_BUCKET_PREFIX-*",
                "arn:aws:s3:::$S3_BUCKET_PREFIX-*/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "fsx:DescribeFileSystems",
                "fsx:DescribeMountTargets",
                "fsx:CreateFileSystem",
                "fsx:DeleteFileSystem"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceAttribute",
                "ec2:DescribeInstanceStatus",
                "ec2:DescribeImages"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
EOF
        
        aws iam put-role-policy --role-name $IAM_ROLE_NAME --policy-name RayMetagenomicsPolicy --policy-document file:///tmp/permissions-policy.json
        
        # Create instance profile
        aws iam create-instance-profile --instance-profile-name $IAM_PROFILE_NAME
        aws iam add-role-to-instance-profile --instance-profile-name $IAM_PROFILE_NAME --role-name $IAM_ROLE_NAME
        
        log_success "IAM role and instance profile created"
    else
        log_info "IAM role already exists: $IAM_ROLE_NAME"
    fi
    
    # Create S3 bucket
    S3_BUCKET="$S3_BUCKET_PREFIX-$(whoami)-$(date +%s)"
    if ! aws s3 ls s3://$S3_BUCKET &>/dev/null; then
        log_info "Creating S3 bucket: $S3_BUCKET"
        
        if [ "$AWS_REGION" == "us-east-1" ]; then
            aws s3 mb s3://$S3_BUCKET
        else
            aws s3 mb s3://$S3_BUCKET --region $AWS_REGION
        fi
        
        log_success "S3 bucket created: $S3_BUCKET"
    else
        log_info "S3 bucket already exists: $S3_BUCKET"
    fi
    
    # Save resource information
    cat > aws-resources.txt << EOF
# AWS Resources Created for Ray Metagenomic Cluster
VPC_ID=$VPC_ID
SUBNET_ID=$SUBNET_ID
SECURITY_GROUP_ID=$SECURITY_GROUP_ID
KEY_PAIR_NAME=$KEY_PAIR_NAME
IAM_ROLE_NAME=$IAM_ROLE_NAME
IAM_PROFILE_NAME=$IAM_PROFILE_NAME
S3_BUCKET=$S3_BUCKET
AWS_REGION=$AWS_REGION
EOF
    
    log_success "AWS resources created and saved to aws-resources.txt"
}

# Get latest Deep Learning AMI
get_latest_ami() {
    log_info "Finding latest Deep Learning AMI..."
    
    # Get the latest Deep Learning AMI for the region
    AMI_ID=$(aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning AMI (Ubuntu 18.04) Version*" \
              "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)
    
    if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
        log_warning "Could not find Deep Learning AMI, using default Ubuntu AMI"
        AMI_ID=$(aws ec2 describe-images \
            --owners 099720109477 \
            --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-20.04-amd64-server-*" \
                  "Name=state,Values=available" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text)
    fi
    
    echo $AMI_ID
}

# Create cluster configuration
create_cluster_config() {
    log_info "Creating cluster configuration..."
    
    # Source AWS resources
    source aws-resources.txt
    
    # Get latest AMI
    AMI_ID=$(get_latest_ami)
    log_info "Using AMI: $AMI_ID"
    
    # Create the cluster configuration by updating the existing template
    sed -i.bak \
        -e "s/ami-0abcdef1234567890/$AMI_ID/g" \
        -e "s/ray-autoscaler-keypair/$KEY_PAIR_NAME/g" \
        -e "s/sg-12345678/$SECURITY_GROUP_ID/g" \
        -e "s/subnet-12345678/$SUBNET_ID/g" \
        -e "s/RayMetagenomicsProfile/$IAM_PROFILE_NAME/g" \
        -e "s/us-east-1/$AWS_REGION/g" \
        metagenomics-autoscaler.yaml
    
    log_success "Cluster configuration updated with your AWS resources"
}

# Create CloudWatch configuration
create_cloudwatch_config() {
    log_info "Creating CloudWatch monitoring configuration..."
    
    mkdir -p cloudwatch
    
    # CloudWatch dashboard configuration
    cat > cloudwatch/dashboard-config.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "Ray/MetagenomicsCluster", "cpu_usage_active", "InstanceId", "ALL" ],
                    [ ".", "mem_used_percent", ".", "." ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "$AWS_REGION",
                "title": "Cluster Resource Utilization"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "Ray/MetagenomicsCluster", "utilization_gpu", "InstanceId", "ALL" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "$AWS_REGION",
                "title": "GPU Utilization"
            }
        }
    ]
}
EOF
    
    log_success "CloudWatch configuration created"
}

# Setup Ray cluster manager script
setup_cluster_manager() {
    log_info "Setting up cluster manager script..."
    
    # Make cluster manager executable
    chmod +x ray_cluster_manager.sh
    
    # Create convenient aliases
    cat > ray_aliases.sh << 'EOF'
#!/bin/bash
# Ray cluster management aliases

alias ray-launch='./ray_cluster_manager.sh launch'
alias ray-status='./ray_cluster_manager.sh status'
alias ray-connect='./ray_cluster_manager.sh connect'
alias ray-dashboard='./ray_cluster_manager.sh dashboard'
alias ray-logs='./ray_cluster_manager.sh logs'
alias ray-teardown='./ray_cluster_manager.sh teardown'
alias ray-metagenomics='./ray_cluster_manager.sh metagenomics'

echo "Ray cluster management aliases loaded:"
echo "  ray-launch      - Launch the cluster"
echo "  ray-status      - Show cluster status"
echo "  ray-connect     - Connect to head node"
echo "  ray-dashboard   - Open dashboard"
echo "  ray-logs        - Show cluster logs"
echo "  ray-teardown    - Teardown cluster"
echo "  ray-metagenomics - Run metagenomic job"
EOF
    
    chmod +x ray_aliases.sh
    
    log_success "Cluster manager and aliases created"
}

# Run validation tests
run_validation() {
    log_info "Running validation tests..."
    
    # Test Ray installation
    python3 -c "import ray; print(f'Ray version: {ray.__version__}')" || {
        log_error "Ray import failed"
        exit 1
    }
    
    # Test AWS connectivity
    aws sts get-caller-identity &>/dev/null || {
        log_error "AWS credentials not working"
        exit 1
    }
    
    # Test boto3
    python3 -c "import boto3; print('boto3 OK')" || {
        log_error "boto3 import failed"
        exit 1
    }
    
    # Validate cluster config
    ./ray_cluster_manager.sh validate || {
        log_error "Cluster configuration validation failed"
        exit 1
    }
    
    log_success "All validation tests passed"
}

# Show next steps
show_next_steps() {
    cat << EOF

${GREEN}========================================${NC}
${GREEN}Setup completed successfully!${NC}
${GREEN}========================================${NC}

${BLUE}AWS Resources Created:${NC}
$(cat aws-resources.txt | grep -E "^[A-Z_]+" | sed 's/^/  /')

${BLUE}Next Steps:${NC}

1. ${YELLOW}Load Ray aliases:${NC}
   source ray_aliases.sh

2. ${YELLOW}Launch the cluster:${NC}
   ray-launch
   # OR
   ./ray_cluster_manager.sh launch

3. ${YELLOW}Submit a metagenomic job:${NC}
   ray-metagenomics SRR1000001 50000 15 100
   # OR
   ./ray_cluster_manager.sh metagenomics SRR1000001 50000 15 100

4. ${YELLOW}Monitor the cluster:${NC}
   ray-dashboard
   ray-status

5. ${YELLOW}When finished, teardown the cluster:${NC}
   ray-teardown

${BLUE}Files Created:${NC}
  metagenomics-autoscaler.yaml  - Ray cluster configuration
  ray_cluster_manager.sh        - Cluster management script
  cloudwatch/                   - CloudWatch monitoring config
  aws-resources.txt             - AWS resource details
  ray_aliases.sh                - Convenient command aliases

${BLUE}Useful Commands:${NC}
  ./ray_cluster_manager.sh help          - Show all available commands
  ./ray_cluster_manager.sh status        - Check cluster status
  ./ray_cluster_manager.sh logs          - View cluster logs
  ./ray_cluster_manager.sh scale 4       - Scale to 4 workers
  ./ray_cluster_manager.sh costs         - Show cost estimates

${YELLOW}Important Notes:${NC}
- Remember to teardown the cluster when not in use to avoid charges
- The cluster uses spot instances by default for cost savings
- Logs are saved in the ./logs directory
- Configuration files have been updated with your AWS resources

${GREEN}Happy clustering! 🧬${NC}
EOF
}

# Main setup function
main() {
    echo "Ray Autoscaler Setup for Metagenomic Clustering"
    echo "==============================================="
    echo ""
    
    case "${1:-all}" in
        "prereq")
            check_prerequisites
            ;;
        "python")
            install_python_dependencies
            ;;
        "aws")
            configure_aws
            create_aws_resources
            ;;
        "config")
            create_cluster_config
            create_cloudwatch_config
            setup_cluster_manager
            ;;
        "validate")
            run_validation
            ;;
        "all")
            check_prerequisites
            install_python_dependencies
            configure_aws
            create_aws_resources
            create_cluster_config
            create_cloudwatch_config
            setup_cluster_manager
            run_validation
            show_next_steps
            ;;
        "help")
            echo "Usage: $0 [step]"
            echo ""
            echo "Steps:"
            echo "  prereq    - Check prerequisites"
            echo "  python    - Install Python dependencies"
            echo "  aws       - Configure AWS and create resources"
            echo "  config    - Create configuration files"
            echo "  validate  - Run validation tests"
            echo "  all       - Run all steps (default)"
            echo "  help      - Show this help"
            ;;
        *)
            log_error "Unknown step: $1"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
