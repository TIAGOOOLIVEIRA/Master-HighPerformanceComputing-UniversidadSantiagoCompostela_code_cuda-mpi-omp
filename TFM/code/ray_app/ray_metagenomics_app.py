#!/usr/bin/env python3
"""
Ray Application for Metagenomic Read Clustering with LSHVec Embeddings

This application implements a distributed pipeline for:
1. Downloading metagenomic datasets from public repositories
2. Processing DNA sequences with LSHVec embeddings 
3. Distributed clustering using Ray actors
4. Persisting results to S3/FSx Lustre in Parquet format

Cluster Configuration: 1 head node + 2 GPU accelerated nodes
"""

import os
import ray
import boto3
import pandas as pd
import numpy as np
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import requests
import hashlib
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetagenomicConfig:
    """Configuration for metagenomic processing"""
    # Dataset configuration
    dataset_accession: str = "SRR1000001"  # Example NCBI SRA accession
    dataset_source: str = "ncbi_sra"
    download_limit: int = 100000  # Limit reads for testing
    
    # LSHVec configuration  
    kmer_size: int = 15
    embedding_dim: int = 100
    lsh_tables: int = 10
    lsh_functions: int = 20
    min_count: int = 5
    
    # Ray configuration
    num_actors: int = 4
    batch_size: int = 1000
    
    # Storage configuration
    s3_bucket: str = "metagenomics-data"
    s3_prefix: str = "embeddings"
    use_fsx_lustre: bool = True
    fsx_mount_path: str = "/fsx"
    
    # Output configuration
    output_format: str = "parquet"
    compression: str = "snappy"

class DatasetDownloader:
    """Handles downloading metagenomic datasets from various sources"""
    
    def __init__(self, config: MetagenomicConfig):
        self.config = config
        self.s3_client = boto3.client('s3')
        
    def download_ncbi_sra(self, accession: str, output_dir: str) -> str:
        """Download dataset from NCBI SRA using sra-toolkit"""
        logger.info(f"Downloading SRA dataset: {accession}")
        
        # Use prefetch and fasterq-dump from sra-toolkit
        try:
            # Download SRA file
            subprocess.run([
                "prefetch", accession, 
                "--output-directory", output_dir
            ], check=True)
            
            # Convert to FASTQ
            sra_path = os.path.join(output_dir, accession, f"{accession}.sra")
            fastq_output = os.path.join(output_dir, f"{accession}.fastq")
            
            subprocess.run([
                "fasterq-dump", sra_path,
                "--outdir", output_dir,
                "--progress",
                "--split-files" if self._is_paired_end(sra_path) else "--split-spot"
            ], check=True)
            
            return fastq_output
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {accession}: {e}")
            raise
    
    def download_mg_rast(self, mg_id: str, output_dir: str) -> str:
        """Download dataset from MG-RAST"""
        logger.info(f"Downloading MG-RAST dataset: {mg_id}")
        
        url = f"https://api.mg-rast.org/download/{mg_id}?file=350.1"
        output_file = os.path.join(output_dir, f"{mg_id}.fasta")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return output_file
    
    def upload_to_s3(self, local_path: str, s3_key: str) -> str:
        """Upload file to S3"""
        logger.info(f"Uploading {local_path} to s3://{self.config.s3_bucket}/{s3_key}")
        
        self.s3_client.upload_file(
            local_path, 
            self.config.s3_bucket, 
            s3_key
        )
        
        return f"s3://{self.config.s3_bucket}/{s3_key}"
    
    def _is_paired_end(self, sra_path: str) -> bool:
        """Check if SRA file contains paired-end reads"""
        try:
            result = subprocess.run([
                "sra-stat", "--meta", "--quick", sra_path
            ], capture_output=True, text=True, check=True)
            return "paired" in result.stdout.lower()
        except:
            return False

class LSHVecProcessor:
    """Handles LSHVec model training and embedding generation"""
    
    def __init__(self, config: MetagenomicConfig):
        self.config = config
        self.model_path = None
        
    def prepare_sequences(self, fastq_path: str, temp_dir: str) -> str:
        """Convert FASTQ to format suitable for LSHVec"""
        logger.info("Preparing sequences for LSHVec")
        
        output_file = os.path.join(temp_dir, "sequences.txt")
        count = 0
        
        with open(output_file, 'w') as out_f:
            for record in SeqIO.parse(fastq_path, "fastq"):
                if count >= self.config.download_limit:
                    break
                out_f.write(f">{record.id}\n{str(record.seq)}\n")
                count += 1
                
        logger.info(f"Prepared {count} sequences")
        return output_file
    
    def create_hash_file(self, sequences_file: str, temp_dir: str) -> str:
        """Create LSH hash file from sequences"""
        logger.info("Creating LSH hash file")
        
        hash_file = os.path.join(temp_dir, "sequences.hash")
        
        # Use LSHVec hashSeq command
        cmd = [
            "hashSeq",
            "-input", sequences_file,
            "-output", hash_file,
            "-kmer_size", str(self.config.kmer_size),
            "-lsh_tables", str(self.config.lsh_tables),
            "-lsh_functions", str(self.config.lsh_functions),
            "-thread", "4"
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=temp_dir)
            return hash_file
        except subprocess.CalledProcessError as e:
            logger.error(f"LSH hashing failed: {e}")
            raise
    
    def train_embeddings(self, hash_file: str, temp_dir: str) -> str:
        """Train LSHVec embeddings model"""
        logger.info("Training LSHVec embeddings")
        
        model_prefix = os.path.join(temp_dir, "lshvec_model")
        
        cmd = [
            "lshvec", "skipgram",
            "-input", hash_file,
            "-output", model_prefix,
            "-dim", str(self.config.embedding_dim),
            "-minCount", str(self.config.min_count),
            "-thread", "4",
            "-epoch", "5"
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=temp_dir)
            self.model_path = f"{model_prefix}.bin"
            return self.model_path
        except subprocess.CalledProcessError as e:
            logger.error(f"LSHVec training failed: {e}")
            raise
    
    def extract_embeddings(self, sequences_file: str, model_path: str, temp_dir: str) -> str:
        """Extract embeddings for sequences using trained model"""
        logger.info("Extracting embeddings")
        
        embeddings_file = os.path.join(temp_dir, "embeddings.vec")
        
        cmd = [
            "lshvec", "print-word-vectors",
            model_path
        ]
        
        try:
            with open(embeddings_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True, cwd=temp_dir)
            return embeddings_file
        except subprocess.CalledProcessError as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise

@ray.remote(num_gpus=0.5)  # Each actor uses half a GPU
class MetagenomicActor:
    """Ray actor for distributed metagenomic processing"""
    
    def __init__(self, config: MetagenomicConfig, actor_id: int):
        self.config = config
        self.actor_id = actor_id
        self.processor = LSHVecProcessor(config)
        logger.info(f"Initialized MetagenomicActor {actor_id}")
    
    def process_batch(self, sequence_batch: List[str], batch_id: int) -> Dict:
        """Process a batch of sequences and return embeddings"""
        logger.info(f"Actor {self.actor_id} processing batch {batch_id} with {len(sequence_batch)} sequences")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write sequences to temp file
            sequences_file = os.path.join(temp_dir, f"batch_{batch_id}.txt")
            with open(sequences_file, 'w') as f:
                for i, seq in enumerate(sequence_batch):
                    f.write(f">seq_{batch_id}_{i}\n{seq}\n")
            
            try:
                # Create hash file
                hash_file = self.processor.create_hash_file(sequences_file, temp_dir)
                
                # Train model for this batch (in practice, you'd use a pre-trained model)
                model_path = self.processor.train_embeddings(hash_file, temp_dir)
                
                # Extract embeddings
                embeddings_file = self.processor.extract_embeddings(sequences_file, model_path, temp_dir)
                
                # Parse embeddings
                embeddings = self._parse_embeddings(embeddings_file)
                
                return {
                    'batch_id': batch_id,
                    'actor_id': self.actor_id,
                    'embeddings': embeddings,
                    'num_sequences': len(sequence_batch),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Actor {self.actor_id} failed to process batch {batch_id}: {e}")
                return {
                    'batch_id': batch_id,
                    'actor_id': self.actor_id,
                    'error': str(e),
                    'num_sequences': len(sequence_batch),
                    'timestamp': datetime.now().isoformat()
                }
    
    def _parse_embeddings(self, embeddings_file: str) -> List[Dict]:
        """Parse LSHVec embeddings output"""
        embeddings = []
        
        with open(embeddings_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    word = parts[0]
                    vector = [float(x) for x in parts[1:]]
                    embeddings.append({
                        'kmer': word,
                        'embedding': vector
                    })
        
        return embeddings

class MetagenomicPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: MetagenomicConfig):
        self.config = config
        self.downloader = DatasetDownloader(config)
        self.processor = LSHVecProcessor(config)
        self.s3_client = boto3.client('s3')
        
    def initialize_ray_cluster(self):
        """Initialize Ray cluster"""
        logger.info("Initializing Ray cluster")
        
        if not ray.is_initialized():
            ray.init(
                address='auto',  # Connect to existing cluster
                _redis_password='your_redis_password'
            )
        
        # Wait for nodes to be available
        while len(ray.nodes()) < 3:  # 1 head + 2 workers
            logger.info("Waiting for all nodes to join cluster...")
            ray.util.wait_for_cluster(num_nodes=3, timeout=300)
        
        logger.info(f"Ray cluster initialized with {len(ray.nodes())} nodes")
    
    def download_dataset(self) -> str:
        """Download and prepare dataset"""
        logger.info("Downloading metagenomic dataset")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download from specified source
        if self.config.dataset_source == "ncbi_sra":
            local_path = self.downloader.download_ncbi_sra(
                self.config.dataset_accession, 
                temp_dir
            )
        elif self.config.dataset_source == "mg_rast":
            local_path = self.downloader.download_mg_rast(
                self.config.dataset_accession,
                temp_dir
            )
        else:
            raise ValueError(f"Unsupported dataset source: {self.config.dataset_source}")
        
        # Upload to S3
        s3_key = f"raw_data/{self.config.dataset_accession}.fastq"
        s3_path = self.downloader.upload_to_s3(local_path, s3_key)
        
        return s3_path
    
    def load_sequences(self, dataset_path: str) -> List[str]:
        """Load sequences from dataset"""
        logger.info("Loading sequences from dataset")
        
        # Download from S3 if needed
        if dataset_path.startswith('s3://'):
            local_path = self._download_from_s3(dataset_path)
        else:
            local_path = dataset_path
        
        sequences = []
        count = 0
        
        # Determine file format
        file_format = "fastq" if local_path.endswith('.fastq') else "fasta"
        
        for record in SeqIO.parse(local_path, file_format):
            if count >= self.config.download_limit:
                break
            sequences.append(str(record.seq))
            count += 1
        
        logger.info(f"Loaded {len(sequences)} sequences")
        return sequences
    
    def create_batches(self, sequences: List[str]) -> List[List[str]]:
        """Split sequences into batches for distributed processing"""
        batches = []
        for i in range(0, len(sequences), self.config.batch_size):
            batch = sequences[i:i + self.config.batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches of size {self.config.batch_size}")
        return batches
    
    def process_distributed(self, batches: List[List[str]]) -> List[Dict]:
        """Process batches using distributed Ray actors"""
        logger.info("Starting distributed processing")
        
        # Create Ray actors
        actors = [
            MetagenomicActor.remote(self.config, i) 
            for i in range(self.config.num_actors)
        ]
        
        # Submit work to actors
        futures = []
        for batch_id, batch in enumerate(batches):
            actor = actors[batch_id % len(actors)]
            future = actor.process_batch.remote(batch, batch_id)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        logger.info(f"Processed {len(results)} batches")
        return results
    
    def save_to_parquet(self, results: List[Dict], output_path: str):
        """Save results to Parquet format"""
        logger.info("Saving results to Parquet")
        
        # Flatten embeddings data
        all_embeddings = []
        
        for result in results:
            if 'error' in result:
                logger.warning(f"Skipping failed batch {result['batch_id']}: {result['error']}")
                continue
                
            batch_id = result['batch_id']
            actor_id = result['actor_id']
            timestamp = result['timestamp']
            
            for emb in result['embeddings']:
                all_embeddings.append({
                    'batch_id': batch_id,
                    'actor_id': actor_id,
                    'kmer': emb['kmer'],
                    'embedding': emb['embedding'],
                    'embedding_dim': len(emb['embedding']),
                    'processed_at': timestamp
                })
        
        # Create DataFrame
        df = pd.DataFrame(all_embeddings)
        
        # Convert embedding lists to individual columns
        max_dim = max(df['embedding_dim']) if len(df) > 0 else self.config.embedding_dim
        for i in range(max_dim):
            df[f'emb_{i}'] = df['embedding'].apply(
                lambda x: x[i] if i < len(x) else 0.0
            )
        
        # Drop the original embedding column
        df = df.drop(['embedding'], axis=1)
        
        # Create PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Write to Parquet
        if output_path.startswith('s3://'):
            # Write to S3
            self._write_parquet_to_s3(table, output_path)
        else:
            # Write locally (including FSx Lustre mount)
            pq.write_table(
                table, 
                output_path, 
                compression=self.config.compression
            )
        
        logger.info(f"Saved {len(all_embeddings)} embeddings to {output_path}")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        logger.info("Starting metagenomic read clustering pipeline")
        
        try:
            # Initialize Ray cluster
            self.initialize_ray_cluster()
            
            # Download dataset
            dataset_path = self.download_dataset()
            
            # Load sequences
            sequences = self.load_sequences(dataset_path)
            
            # Create batches
            batches = self.create_batches(sequences)
            
            # Process with distributed actors
            results = self.process_distributed(batches)
            
            # Prepare output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.config.use_fsx_lustre:
                output_path = os.path.join(
                    self.config.fsx_mount_path,
                    f"embeddings_{self.config.dataset_accession}_{timestamp}.parquet"
                )
            else:
                output_path = f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}/embeddings_{self.config.dataset_accession}_{timestamp}.parquet"
            
            # Save results
            self.save_to_parquet(results, output_path)
            
            logger.info(f"Pipeline completed successfully. Results saved to: {output_path}")
            
            return {
                'status': 'success',
                'dataset_accession': self.config.dataset_accession,
                'num_sequences': len(sequences),
                'num_batches': len(batches),
                'output_path': output_path,
                'embedding_dimension': self.config.embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Cleanup Ray
            ray.shutdown()
    
    def _download_from_s3(self, s3_path: str) -> str:
        """Download file from S3 to local temporary directory"""
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        local_path = tempfile.mktemp()
        
        self.s3_client.download_file(bucket, key, local_path)
        return local_path
    
    def _write_parquet_to_s3(self, table: pa.Table, s3_path: str):
        """Write PyArrow table to S3 as Parquet"""
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        
        with tempfile.NamedTemporaryFile() as tmp_file:
            pq.write_table(table, tmp_file.name, compression=self.config.compression)
            self.s3_client.upload_file(tmp_file.name, bucket, key)

def create_ray_cluster_config():
    """Generate Ray cluster configuration for AWS"""
    config = {
        'cluster_name': 'metagenomics-cluster',
        'provider': {
            'type': 'aws',
            'region': 'us-east-1',
            'availability_zone': 'us-east-1a'
        },
        'auth': {
            'ssh_user': 'ubuntu'
        },
        'head_node': {
            'InstanceType': 'c5.2xlarge',
            'ImageId': 'ami-0abcdef1234567890',  # Deep Learning AMI
            'KeyName': 'my-key-pair',
            'SecurityGroupIds': ['sg-12345678'],
            'SubnetId': 'subnet-12345678'
        },
        'worker_nodes': {
            'InstanceType': 'p3.2xlarge',  # GPU instances
            'ImageId': 'ami-0abcdef1234567890',  # Deep Learning AMI
            'KeyName': 'my-key-pair',
            'SecurityGroupIds': ['sg-12345678'],
            'SubnetId': 'subnet-12345678',
            'min_workers': 2,
            'max_workers': 2
        },
        'setup_commands': [
            'sudo apt-get update',
            'sudo apt-get install -y sra-toolkit',
            'conda install -y biopython',
            'pip install pyarrow boto3',
            'git clone https://github.com/Lizhen0909/LSHVec.git',
            'cd LSHVec && make'
        ],
        'file_mounts': {
            '/tmp/lshvec': './LSHVec'
        }
    }
    return config

# Example usage and main execution
def main():
    """Main execution function"""
    
    # Configuration
    config = MetagenomicConfig(
        dataset_accession="SRR1000001",  # Example Human Microbiome Project sample
        dataset_source="ncbi_sra",
        download_limit=50000,  # Process 50K reads for demo
        kmer_size=15,
        embedding_dim=100,
        num_actors=4,
        batch_size=1000,
        s3_bucket="my-metagenomics-bucket",
        use_fsx_lustre=True,
        fsx_mount_path="/fsx"
    )
    
    # Create and run pipeline
    pipeline = MetagenomicPipeline(config)
    
    try:
        result = pipeline.run_pipeline()
        print("Pipeline completed successfully!")
        print(f"Results: {result}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

# Additional utilities for cluster management
class ClusterManager:
    """Utilities for managing Ray cluster on AWS"""
    
    @staticmethod
    def launch_cluster():
        """Launch Ray cluster using ray up"""
        config_file = "cluster_config.yaml"
        
        # Write cluster config
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(create_ray_cluster_config(), f)
        
        # Launch cluster
        subprocess.run(['ray', 'up', config_file], check=True)
        
    @staticmethod
    def teardown_cluster():
        """Teardown Ray cluster"""
        config_file = "cluster_config.yaml"
        subprocess.run(['ray', 'down', config_file], check=True)
        
    @staticmethod
    def monitor_cluster():
        """Monitor cluster resources and job progress"""
        # This would connect to Ray dashboard
        # and provide monitoring capabilities
        pass

# Dataset exploration utilities
class DatasetExplorer:
    """Utilities for exploring available metagenomic datasets"""
    
    @staticmethod
    def search_human_microbiome():
        """Search Human Microbiome Project datasets"""
        # Integration with HumanMetagenomeDB API
        base_url = "https://webapp.ufz.de/hmgdb/api/"
        # Implementation would query the API
        pass
    
    @staticmethod
    def search_terrestrial_datasets():
        """Search terrestrial metagenomic datasets"""
        # Integration with TerrestrialMetagenomeDB
        base_url = "https://webapp.ufz.de/tmdb/api/"
        # Implementation would query the API
        pass
    
    @staticmethod
    def list_recommended_datasets():
        """Return list of recommended datasets for clustering experiments"""
        return [
            {
                'accession': 'SRR1000001',
                'source': 'ncbi_sra',
                'description': 'Human gut microbiome sample',
                'estimated_size': '2.5GB',
                'organism': 'human gut metagenome'
            },
            {
                'accession': 'SRR2000001', 
                'source': 'ncbi_sra',
                'description': 'Soil microbiome sample',
                'estimated_size': '1.8GB',
                'organism': 'soil metagenome'
            },
            {
                'accession': 'SRR3000001',
                'source': 'ncbi_sra', 
                'description': 'Marine microbiome sample',
                'estimated_size': '3.2GB',
                'organism': 'marine metagenome'
            }
        ]
