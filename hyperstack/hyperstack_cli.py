import argparse
import datetime
import json
import os

import requests
from dotenv import load_dotenv

commands = {}


def _register(func):
    commands[func.__name__] = func
    return func


def _request(method, url, headers={}, json=None):
    url = f"https://infrahub-api.nexgencloud.com/v1/core/{url}"
    headers["api_key"] = os.getenv("HYPERSTACK_API_KEY")
    headers["accept"] = "application/json"
    response = requests.request(method, url, headers=headers, json=json)
    return response.json()


@_register
def create_vm(cloud_init, recipe, model, gpu):
    """Create a VM."""
    flavor_name = {
        "a100": "n3-A100x1",
        "a6000": "RTX-A6000",
    }[gpu]
    name = f"VM-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    data = {
        "name": name,
        "environment_name": "default-CANADA-1",
        "image_name": "Ubuntu Server 22.04 LTS R535 CUDA 12.2",
        "flavor_name": flavor_name,
        "key_name": "DTU Workstation Canada",  # change to your key name in dashboard
        "assign_floating_ip": True,
        "count": 1,
    }
    with open(cloud_init, "r") as f:
        cloud_init = f.read()
    for env_var in [
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HYPERSTACK_API_KEY",
        "GITLAB_ACCESS_TOKEN",
    ]:
        if os.getenv(env_var) is None:
            raise ValueError(f"Environment variable {env_var} is not set.")
        assert f"${{{env_var}}}" in cloud_init
        cloud_init = cloud_init.replace(f"${{{env_var}}}", os.getenv(env_var))
    assert "${HYPERSTACK_VM_NAME}" in cloud_init
    cloud_init = cloud_init.replace("${HYPERSTACK_VM_NAME}", name)
    assert "${MBCHL_RECIPE}" in cloud_init
    cloud_init = cloud_init.replace("${MBCHL_RECIPE}", recipe)
    assert "${MBCHL_MODEL}" in cloud_init
    cloud_init = cloud_init.replace("${MBCHL_MODEL}", model)
    data["user_data"] = cloud_init
    return _request("POST", "virtual-machines", json=data)


@_register
def delete_all_vms():
    """Delete all VMs."""
    vms = list_vms()
    return [delete_vm(vm["id"]) for vm in vms["instances"]]


@_register
def delete_vm(vm_id):
    """Delete a VM."""
    return _request("DELETE", f"virtual-machines/{vm_id}")


@_register
def delete_vm_by_name(name):
    """Delete a VM by name."""
    vm = get_vm_by_name(name)
    if vm is not None:
        return delete_vm(vm["id"])
    return None


@_register
def enable_ssh(vm_id):
    """Enable SSH access to a VM."""
    data = {
        "remote_ip_prefix": "0.0.0.0/0",
        "direction": "ingress",
        "ethertype": "IPv4",
        "protocol": "tcp",
        "port_range_min": 22,
        "port_range_max": 22,
    }
    return _request("POST", f"virtual-machines/{vm_id}/sg-rules", json=data)


@_register
def enable_ssh_all():
    """Enable SSH access to all VMs."""
    vms = list_vms()
    return [enable_ssh(vm["id"]) for vm in vms["instances"]]


@_register
def get_balance():
    """Get credit balance."""
    url = "https://infrahub-api.nexgencloud.com/v1/billing/user-credit/credit"
    headers = {
        "api_key": os.getenv("HYPERSTACK_API_KEY"),
        "accept": "application/json",
    }
    response = requests.get(url, headers=headers)
    return response.json()


@_register
def get_flavors():
    """Get flavors."""
    return _request("GET", "flavors")


@_register
def get_pricebook():
    """Get pricebook."""
    url = "https://infrahub-api.nexgencloud.com/v1/pricebook"
    headers = {
        "api_key": os.getenv("HYPERSTACK_API_KEY"),
        "accept": "application/json",
    }
    response = requests.get(url, headers=headers)
    return response.json()


@_register
def get_vm_by_name(name):
    """Get a VM by name."""
    vms = list_vms()
    for vm in vms["instances"]:
        if vm["name"] == name:
            return vm
    return None


@_register
def list_vms():
    """List VMs."""
    return _request("GET", "virtual-machines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=commands.keys())
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()

    load_dotenv()

    out = commands[args.command](*args.args)
    print(json.dumps(out, indent=4))
