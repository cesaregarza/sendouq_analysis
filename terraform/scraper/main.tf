terraform {
    required_providers {
        digitalocean = {
            source = "digitalocean/digitalocean"
            version = "~> 2.0"
        }
    }
}

variable "do_token" {}
variable "pvt_key" {}

provider "digitalocean" {
    token = var.do_token
}

data "digitalocean_ssh_key" "github_actions" {
    name = "github_actions"
}

data "digitalocean_ssh_key" "wsl" {
    name = "wsl"
}
