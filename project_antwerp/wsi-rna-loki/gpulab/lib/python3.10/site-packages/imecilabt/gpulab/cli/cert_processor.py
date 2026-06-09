import re
import datetime
from typing import List

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import ExtensionOID
from imecilabt_utils import URN
from pydantic import BaseModel

from imecilabt.gpulab.util import gpulab_logging

logger = gpulab_logging.getLogger("cert_processor")


class CertInfo(BaseModel):
    subject_user_urn: str
    subject_username: str
    not_valid_after: datetime.datetime
    not_valid_before: datetime.datetime
    emails: List[str]


def process_cert(certfile: str) -> CertInfo:
    with open(certfile, 'r') as f:
        cert_content = f.read()  # encode because load_pem_x509_certificate needs bytes
    try:
        cert_begin_delim = '-----BEGIN CERTIFICATE-----'
        cert_end_delim = '-----END CERTIFICATE-----'
        cert_content = re.findall(r"{begin}.*?{end}".format(begin=cert_begin_delim, end=cert_end_delim),
                                   cert_content, re.DOTALL)[0]

        cert = x509.load_pem_x509_certificate(cert_content.encode('utf-8'), default_backend())
    except ValueError:
        logger.exception(f'Problem parsing PEM: {cert_content!r}')
        raise
    try:
        ext = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        urns = ext.value.get_values_for_type(x509.UniformResourceIdentifier)
        emails = ext.value.get_values_for_type(x509.RFC822Name) or []
    except ValueError:
        logger.exception(f'Problem processing certificate: {cert_content!r}')
        raise
    user_urn = str(urns[0])
    username = URN(urn=user_urn).name
    return CertInfo(
        subject_user_urn=user_urn,
        subject_username=username,
        not_valid_after=cert.not_valid_after.astimezone(datetime.timezone.utc),
        not_valid_before=cert.not_valid_before.astimezone(datetime.timezone.utc),
        emails=emails
    )
