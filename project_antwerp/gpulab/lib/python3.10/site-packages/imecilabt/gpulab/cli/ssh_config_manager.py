import os
from os.path import expanduser
from socket import AF_INET, SOCK_STREAM, socket
from textwrap import dedent
from typing import List

import click
from imecilabt.gpulab.schemas.job2 import Job


def sh_quote_arg(c: str) -> str:
    if "'" not in c:
        if " " not in c and '"' not in c and "$" not in c:
            return c
        else:
            return f"'{c}'"
    else:
        c = c.replace("'", """'"'"'""")
        return f"'{c}'"


def command_array_to_str(command: List[str]) -> str:
    return " ".join([sh_quote_arg(c) for c in command])


def _ensure_ssh_config_include(debug: bool = False) -> str:
    ssh_config_filename = expanduser("~/.ssh/config")
    ssh_config_gpulab_filename_base = "gpulab_hosts_config"
    ssh_config_gpulab_filename = expanduser(f"~/.ssh/{ssh_config_gpulab_filename_base}")

    ssh_dir = expanduser("~/.ssh/")
    if not os.path.exists(ssh_dir):
        os.mkdir(ssh_dir, 0o700)
        click.secho(f"Created {ssh_dir}")

    # Check if Host already added
    must_add_include = True
    if os.path.exists(ssh_config_filename):
        created_ssh_config = False
        with open(ssh_config_filename, "r") as f:
            for line in f:
                if line.lstrip().startswith(
                    f"Include {ssh_config_gpulab_filename_base}"
                ):
                    must_add_include = False
                    break
    else:
        created_ssh_config = True

    if must_add_include:
        with open(ssh_config_filename, "a") as f:
            f.write("\n")
            f.write("Host *\n")
            f.write(f"   Include {ssh_config_gpulab_filename_base}\n")
            f.write("\n")
        if debug:
            if created_ssh_config:
                click.secho(f"Added include to {ssh_config_filename}")
            else:
                click.secho(f"Created {ssh_config_filename}")

    return ssh_config_gpulab_filename


def job_ssh_config_hostname(job: Job, use_proxy: bool) -> str:
    if use_proxy:
        return f"gpulab-{job.short_uuid}-proxy"
    else:
        return f"gpulab-{job.short_uuid}"


def _job_ssh_config_exists(job: Job, ssh_config_gpulab_filename: str):
    if os.path.exists(ssh_config_gpulab_filename):
        with open(ssh_config_gpulab_filename, "r") as f:
            for line in f:
                if line.startswith(
                    f"Host={job_ssh_config_hostname(job, True)}"
                ) or line.startswith(f"Host={job_ssh_config_hostname(job, False)}"):
                    return True
    return False


def set_jobs_ssh_config(login_pem_filename: str, jobs: List[Job], debug: bool = False):
    ssh_config_gpulab_filename = _ensure_ssh_config_include(debug)
    with open(ssh_config_gpulab_filename, "w") as f:
        for job in jobs:
            _actual_write_job_ssh_config(f, login_pem_filename, job, proxy=False)
            _actual_write_job_ssh_config(f, login_pem_filename, job, proxy=True)
    if debug:
        click.secho(f"Rewrote {len(jobs)} jobs in {ssh_config_gpulab_filename}")
    os.chmod(ssh_config_gpulab_filename, 0o700)


def add_job_ssh_config(
    login_pem_filename: str,
    job: Job,
    return_proxy_fake_hostname: bool,
    debug: bool = False,
) -> str:
    ssh_config_gpulab_filename = _ensure_ssh_config_include(debug)
    if _job_ssh_config_exists(job, ssh_config_gpulab_filename):
        return job_ssh_config_hostname(job, return_proxy_fake_hostname)

    created = os.path.exists(ssh_config_gpulab_filename)
    with open(ssh_config_gpulab_filename, "a") as f:
        fake_hostname = _actual_write_job_ssh_config(
            f, login_pem_filename, job, proxy=False
        )
        fake_hostname_proxy = _actual_write_job_ssh_config(
            f, login_pem_filename, job, proxy=True
        )
    if created:
        os.chmod(ssh_config_gpulab_filename, 0o700)
        click.secho(f"Created {ssh_config_gpulab_filename} for job {job.uuid}")
    else:
        click.secho(f"Appended {ssh_config_gpulab_filename} with job {job.uuid}")
    return fake_hostname if not return_proxy_fake_hostname else fake_hostname_proxy


def _actual_write_job_ssh_config(
    f, login_pem_filename: str, job: Job, *, proxy: bool
) -> str | None:
    fake_hostname = job_ssh_config_hostname(job, proxy)

    if not job.state.resources:
        # if job has no resources, we cannot create an SSH config entry
        return None

    if not job.state.resources.ssh_host or not job.state.resources.ssh_username:
        # if job has no SSH info, we cannot create an SSH config entry
        return None

    if (
        proxy
        and job.state.resources.ssh_proxy_host
        and job.state.resources.ssh_proxy_port
        and job.state.resources.ssh_proxy_username
    ):
        proxy_line = (
            f'ProxyCommand ssh -i "{login_pem_filename}" '
            f"-oPort={job.state.resources.ssh_proxy_port} "
            f"{job.state.resources.ssh_proxy_username}@"
            f"{job.state.resources.ssh_proxy_host}"
            " -W %h:%p"
        )
    else:
        proxy_line = ""

    ssh_config_entry = f"""\
    Host {fake_hostname}
        Hostname {job.state.resources.ssh_host}
        Port {job.state.resources.ssh_port if job.state.resources.ssh_port else 22}
        User {job.state.resources.ssh_username}
        {proxy_line}
        IdentityFile "{login_pem_filename}"
        IdentitiesOnly yes
        PubkeyAuthentication yes
        ChallengeResponseAuthentication no
        HostbasedAuthentication no
        GSSAPIAuthentication no
        KbdInteractiveAuthentication no
        PasswordAuthentication no   
        
    """

    f.write(dedent(ssh_config_entry))
    return fake_hostname


def find_free_ssh_tunnel_port(debug: bool = False) -> int:
    start_port = 2222
    port = start_port
    while port < start_port + 42:
        with socket(AF_INET, SOCK_STREAM) as s:
            free = s.connect_ex(("localhost", port)) != 0
            if debug:
                click.echo(f"port {port} is free: {free}")
            if free:
                return port
        port += 1
    raise Exception("Could not find a free port for SSH tunnel.")
