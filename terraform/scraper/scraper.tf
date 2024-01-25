variable "postgres_user" {}
variable "postgres_password" {}
variable "postgres_db" {}
variable "postgres_host" {}
variable "postgres_port" {}

resource "digitalocean_droplet" "sendouq_scraper" {
    image = "ubuntu-22-04-x64"
    name = "sendouq-scraper"
    region = "nyc3"
    size = "s-1vcpu-1gb"
    ssh_keys = [
        data.digitalocean_ssh_key.github_actions.id,
        data.digitalocean_ssh_key.wsl.id,
    ]
    connection {
        host = self.ipv4_address
        user = "root"
        type = "ssh"
        private_key = var.pvt_key
        timeout = "2m"
    }
    provisioner "remote-exec" {
        inline = [
            "export PATH=$PATH:/usr/bin",
            "export POSTGRES_USER=${var.postgres_user}",
            "export POSTGRES_PASSWORD=${var.postgres_password}",
            "export POSTGRES_DB=${var.postgres_db}",
            "export POSTGRES_HOST=${var.postgres_host}",
            "export POSTGRES_PORT=${var.postgres_port}",
            "export DO_API_TOKEN=${var.do_token}",
            "sudo apt-get update",
            # Kill the process that's using apt, update is hanging for some reason
            "sudo pkill -9 apt-get",
            "sudo apt-get install -y docker.io",
            "docker pull registry.digitalocean.com/sendouq/scraper:latest",
            "docker run -d --name scraper registry.digitalocean.com/sendouq/scraper:latest",
        ]
    }
}
