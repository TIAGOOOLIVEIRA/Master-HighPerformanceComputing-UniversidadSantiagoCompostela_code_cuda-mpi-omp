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
