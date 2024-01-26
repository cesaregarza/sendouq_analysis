variable "source_ip" {}

resource "digitalocean_droplet" "sendouq_scraper" {
    image = "ubuntu-22-04-x64"
    name = "sendouq-scraper"
    region = "nyc3"
    size = "s-1vcpu-1gb"
    ssh_keys = [
        data.digitalocean_ssh_key.github_actions.id,
        data.digitalocean_ssh_key.wsl.id,
        data.digitalocean_ssh_key.github_actions_ed25519.id,
    ]
}

# Allow SSH from source_ip
resource "digitalocean_firewall" "sendouq_scraper" {
    name = "sendouq-scraper"
    droplet_ids = [digitalocean_droplet.sendouq_scraper.id]

    inbound_rule {
        protocol = "tcp"
        port_range = "22"
        source_addresses = [var.source_ip]
    }
}

output "scraper_ip" {
    value = digitalocean_droplet.sendouq_scraper.ipv4_address
}
