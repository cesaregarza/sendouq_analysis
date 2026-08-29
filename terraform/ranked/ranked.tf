terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

variable "source_ip" {}
variable "joy_ip" {}
variable "do_token" {}
variable "droplet_size" {
  description = "Droplet size slug for the ranked worker"
  type        = string
  default     = "c-32"  # CPU-optimized: 32 vCPUs, 64GB RAM (for fast LOO at scale)
  # Alternative options:
  # "c-16"  - 16 vCPUs, 32GB RAM (~$0.286/hr, good for 10-15k players)
  # "c-8"   - 8 vCPUs, 16GB RAM (~$0.143/hr, good for <10k players)
  # "s-1vcpu-2gb" - 1 vCPU, 2GB RAM (~$0.007/hr, original, no LOO support)
}

provider "digitalocean" {
  token = var.do_token
}

module "digitalocean_infra" {
  source   = "../digitalocean_infra"
  do_token = var.do_token
}

resource "digitalocean_tag" "ranked_worker" {
  name = "ranked_worker"
}

resource "digitalocean_droplet" "sendouq_ranked" {
  image    = "ubuntu-22-04-x64"
  name     = "sendouq-ranked"
  region   = "nyc3"
  size     = var.droplet_size
  ssh_keys = module.digitalocean_infra.ssh_key_ids
  tags     = [digitalocean_tag.ranked_worker.name]
}

output "ranked_ip" {
  value = digitalocean_droplet.sendouq_ranked.ipv4_address
}

output "ranked_id" {
  value = digitalocean_droplet.sendouq_ranked.id
}
