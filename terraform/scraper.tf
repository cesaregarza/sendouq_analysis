resource "digitalocean_droplet" "sendouq_scraper" {
    image = "ubuntu-22-04-x64"
    name = "sendouq-scraper"
    region = "nyc3"
    size = "s-1vcpu-1gb"
    ssh_keys = [
        data.digitalocean_ssh_key.terraform.id
    ]
    connection {
        host = self.ipv4_address
        user = "root"
        type = "ssh"
        private_key = file(var.pvt_key)
        timeout = "2m"
    }
    provisioner "remote-exec" {
        inline = [
            "export PATH=$PATH:/usr/bin",
            "sudo apt-get update",
            "sudo apt-get install -y docker.io",
        ]
    }
}