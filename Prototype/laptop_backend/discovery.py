import socket
import asyncio
import os
import psutil
import ipaddress
from generate_ssl import get_all_local_ips, generate_self_signed_cert

DISCOVERY_PORT = 8766
BEACON_INTERVAL = 2  # seconds

def get_broadcast_targets():
    """
    Finds all active non-loopback IPv4 interfaces and calculates their exact subnet broadcast addresses.
    Example: 10.192.200.160/255.255.255.0 -> (IP: 10.192.200.160, Broadcast: 10.192.200.255)
    """
    targets = []
    try:
        for iface_name, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family.name == 'AF_INET':
                    ip = addr.address
                    netmask = addr.netmask
                    # Skip loopback and APIPA (169.254.x.x)
                    if ip.startswith("127.") or ip.startswith("169.254."):
                        continue
                    try:
                        net = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
                        bcast = str(net.broadcast_address)
                        targets.append((ip, bcast))
                    except Exception:
                        targets.append((ip, "255.255.255.255"))
    except Exception as e:
        print(f"[Discovery] Error detecting interfaces: {e}")
        
    return targets

async def start_discovery_beacon(server_port=8765):
    """
    Broadcasts UDP beacons directly to all active subnet broadcast addresses.
    Guarantees that Wi-Fi, Ethernet, and hotspot interfaces deliver beacons to smartwatches.
    """
    print(f"Auto-Discovery Beacon starting on UDP port {DISCOVERY_PORT}...")
    
    if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
        generate_self_signed_cert()
        
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setblocking(False)
    
    while True:
        try:
            targets = get_broadcast_targets()
            for ip, bcast in targets:
                msg = f"DIPSEER_SERVER:{ip}:{server_port}".encode('utf-8')
                try:
                    sock.sendto(msg, (bcast, DISCOVERY_PORT))
                except Exception:
                    pass
                try:
                    sock.sendto(msg, ('255.255.255.255', DISCOVERY_PORT))
                except Exception:
                    pass
        except Exception:
            pass
            
        await asyncio.sleep(BEACON_INTERVAL)
