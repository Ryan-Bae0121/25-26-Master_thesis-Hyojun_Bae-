import re
from urllib.parse import urlparse
from uuid import UUID
from typing import Optional


def is_valid_email(email: Optional[str]) -> bool:
    """
    Basic check if a string is an email address email is valid.
    (Could be extended to check for illegal chars etc.)
    :param email:
    :return: False if the string doesn't represent an email address. True if it probably does.
    """
    if not email:
        return False
    email = str(email.strip())
    if email.count('@') == 1 and '.' in email and len(email) >= 6:
        return True
    return False


ALLOWED_HOST_KEY_ALGOS = [
    # current secure versions according to https://github.com/jtesta/ssh-audit
    'ssh-ed25519-cert-v01@openssh.com,',
    'ssh-rsa-cert-v01@openssh.com',
    'ssh-ed25519',
    'ssh-rsa',
]


def is_valid_ssh_key(ssh_pub_key: Optional[str]) -> bool:
    """
    Checks if a string looks like an OpenSSH public key string. (= the format found in ~/.ssh/id_rsa.pub)
    Only a basic check: this does not decode the SSH public key to check.
    :param ssh_pub_key: a string to check
    :return: False if the string does not contain an OpenSSH public key string. True if it probably does.
    """
    if not ssh_pub_key:
        return False
    ssh_pub_key = str(ssh_pub_key.strip())
    key_parts = ssh_pub_key.split(' ')
    host_algo = key_parts[0]
    if host_algo in ALLOWED_HOST_KEY_ALGOS and len(key_parts) >= 2:
        return True
    return False


def is_valid_uuid4(uuid_str: str) -> bool:
    try:
        uuid_obj = UUID(uuid_str, version=4)
    except ValueError:
        return False

    return str(uuid_obj) == uuid_str


def is_valid_uuid(uuid_str: str) -> bool:
    try:
        uuid_obj = UUID(uuid_str)
    except ValueError:
        return False

    recontructed_uuid_str = str(uuid_obj)
    assert len(recontructed_uuid_str) == 36, "is_valid_uuid({}) uuid_obj={} recontructed_uuid_str={} len={}" \
        .format(uuid_str, uuid_obj, recontructed_uuid_str, len(recontructed_uuid_str) == 36)
    return recontructed_uuid_str.lower() == uuid_str.lower()


URL_HOST_DISALLOWED_CHARS_REGEX = re.compile(r'[^a-zA-Z0-9.-]')


def is_valid_url(url: str):
    """
    Is the url valid?
    This is a fast check, but not a complete one. It will catch obvious problems, but not all of them.
    :param url:
    :return:
    """
    try:
        result = urlparse(url)
        if result.port is not None:
            port = int(result.port)
            if port <= 0 or port > 65535:
                return False
        if result.scheme not in ('https', 'http'):
            return False
        if not result.hostname or bool(URL_HOST_DISALLOWED_CHARS_REGEX.search(result.hostname)):
            return False
        return True
    except ValueError:
        return False
