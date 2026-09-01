import os
import socket
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import ipaddress

def get_all_local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
        
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ":" not in ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
        
    return list(ips) if ips else ["127.0.0.1"]

def get_local_ip():
    ips = get_all_local_ips()
    # Prefer non-127.0.0.1 IP
    for ip in ips:
        if not ip.startswith("127."):
            return ip
    return "127.0.0.1"

def generate_self_signed_cert(cert_file="cert.pem", key_file="key.pem"):
    all_ips = get_all_local_ips()
    primary_ip = get_local_ip()
    print(f"Generating self-signed SSL certificate for IPs: {all_ips} and localhost...")
    
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Generate public certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, u"LK"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u"Western Province"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, u"Colombo"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"DipSEER"),
        x509.NameAttribute(NameOID.COMMON_NAME, primary_ip),
    ])
    
    san_list = [
        x509.DNSName(u"localhost"),
        x509.IPAddress(ipaddress.IPv4Address(u"127.0.0.1"))
    ]
    for ip in all_ips:
        if ip != "127.0.0.1":
            try:
                san_list.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
            except Exception:
                pass
        
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        private_key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.datetime.utcnow()
    ).not_valid_after(
        datetime.datetime.utcnow() + datetime.timedelta(days=365)
    ).add_extension(
        x509.SubjectAlternativeName(san_list),
        critical=False,
    ).sign(private_key, hashes.SHA256())

    # Write private key
    with open(key_file, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Write certificate
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
        
    print(f"Generated {cert_file} and {key_file} successfully for {all_ips}.")

if __name__ == "__main__":
    generate_self_signed_cert()
