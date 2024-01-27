terraform {
    required_providers {
        digitalocean = {
            source = "digitalocean/digitalocean"
            version = "~> 2.0"
        }
    }
}

variable "do_token" {}

provider "digitalocean" {
    token = var.do_token
}

data "digitalocean_database_cluster" "sendouq_db" {
    name = "db-postgresql-nyc3-ceg"
}

data "digitalocean_ssh_key" "github_actions" {
    name = "github_actions"
}

data "digitalocean_ssh_key" "github_actions_ed25519" {
    name = "github_actions_ed25519"
}

data "digitalocean_ssh_key" "wsl" {
    name = "wsl"
}
