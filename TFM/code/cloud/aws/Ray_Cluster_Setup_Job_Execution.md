# Setup a Ray cluster leveraging recipe automation for AWS
	Ray - AWS (from MacOS)
		
		Go to the AWS Console → IAM → Users → select your user (e.g. Admin-at-002458070613).
		Click Security credentials → Create access key.
		Copy the Access key ID (AKIA…) and Secret access key (a 40-character string).

		$ aws configure
			AWS Access Key ID
			AWS Secret Access Key

		$ aws ec2 create-key-pair --key-name ray-macos-key --query 'KeyMaterial' --output text > ~/.ssh/ray-macos-key.pem
		$ chmod 400 ~/.ssh/ray-macos-key.pem


		$ aws sts get-caller-identity


		$ brew install python
		$ python3 -m venv ~/.venvs/ray
		$ source ~/.venvs/ray/bin/activate

		$ wget https://github.com/ray-project/ray/blob/master/python/ray/autoscaler/aws/example-full.yaml
		
        #$ ray up example-full.yaml
        #Customized version with some fixes
        $ ray up ray-cluster-example-full.yaml
