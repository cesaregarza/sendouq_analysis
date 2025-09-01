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
  default     = "s-1vcpu-2gb"
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

resource "digitalocean_database_firewall" "sendouq_ranked" {
  cluster_id = module.digitalocean_infra.database_cluster_id
  rule {
    type  = "tag"
    value = digitalocean_tag.ranked_worker.name
  }
  rule {
    type  = "ip_addr"
    value = var.joy_ip
  }
}

output "ranked_ip" {
  value = digitalocean_droplet.sendouq_ranked.ipv4_address
}

output "ranked_id" {
  value = digitalocean_droplet.sendouq_ranked.id
}
