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

provider "digitalocean" {
  token = var.do_token
}

module "digitalocean_infra" {
  source   = "../digitalocean_infra"
  do_token = var.do_token
}

resource "digitalocean_droplet" "sendouq_agg" {
    image = "ubuntu-22-04-x64"
    name = "sendouq-agg"
    region = "nyc3"
    size = "s-2vcpu-4gb"
    ssh_keys = module.digitalocean_infra.ssh_key_ids
}

resource "digitalocean_database_firewall" "sendouq_agg" {
    cluster_id = module.digitalocean_infra.database_cluster_id
    rule {
        type = "ip_addr"
        value = digitalocean_droplet.sendouq_agg.ipv4_address
    }
    rule {
        type = "ip_addr"
        value = var.joy_ip
    }
}

output "agg_ip" {
    value = digitalocean_droplet.sendouq_agg.ipv4_address
}

output "agg_id" {
    value = digitalocean_droplet.sendouq_agg.id
}
