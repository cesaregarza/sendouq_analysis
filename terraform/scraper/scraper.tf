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
    # user_data = file("cloud-init.yml")
}

resource "digitalocean_database_firewall" "sendouq_scraper" {
    name = "sendouq-scraper"
    database_cluster_id = digitalocean_database_cluster.sendouq_scraper.id
    rules = [
        {
            type = "ip_addr"
            value = digitalocean_droplet.sendouq_scraper.ipv4_address
        },
    ]
}

output "scraper_ip" {
    value = digitalocean_droplet.sendouq_scraper.ipv4_address
}
