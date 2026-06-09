import logging
import re
from subprocess import PIPE, STDOUT, Popen
from typing import Optional
from pathlib import Path


def _call_git(*params: str, **kwargs) -> str:
    res = ''
    com = ['git']
    com.extend(params)

    kwargs['stdout'] = PIPE
    kwargs['stderr'] = STDOUT
    kwargs['universal_newlines'] = True

    # Annoyingly, git doesn't seem to listen to GIT_ASKPASS and SSH_ASKPASS, and use TTY to get pass.
    # Solution: from TTY: use "setsid -w <git command>"
    #           not from TTY: should work
    with open('/dev/null', 'r') as devnull:
        kwargs['stdin'] = devnull
        with Popen(com, **kwargs) as proc:
            for line in proc.stdout:
                res += line
            res_code = proc.wait()
            if res_code != 0:
                raise Exception('Command returned error ({}): {} (command={} )'.format(res_code, res.strip(), com))
    return res.strip()


def _call_sentry_cli(*params: str, **kwargs) -> str:
    res = ''
    com = ['sentry-cli']
    com.extend(params)

    kwargs['stdout'] = PIPE
    kwargs['stderr'] = STDOUT
    kwargs['universal_newlines'] = True

    with Popen(com, **kwargs) as proc:
        for line in proc.stdout:
            res += line
        res_code = proc.wait()
        if res_code != 0:
            raise Exception('Command returned error ({}): {} (command={} )'.format(res_code, res.strip(), com))
    return res.strip()


def _find_git_sha(repo_filename: str) -> Optional[str]:
    source_path = Path(repo_filename).resolve()
    if not source_path:
        return None
    source_dir = source_path.parent
    if not source_dir:
        return None
    try:
        res = _call_git("rev-parse", "HEAD", cwd=source_dir).strip()
        assert re.compile(r'^([0-9a-f]+)$').match(res)
        assert len(res) > 10
        return res
    except:
        logging.error('Failed "git rev-parse head" in "{}" (will ignore)'.format(source_dir), exc_info=True)
        return None


def _find_git_branch(repo_filename: str) -> Optional[str]:
    source_path = Path(repo_filename).resolve()
    if not source_path:
        return None
    source_dir = source_path.parent
    if not source_dir:
        return None
    try:
        res = _call_git("rev-parse", "--abbrev-ref", "HEAD", cwd=source_dir).strip()
        assert re.compile(r'^([^ ]+)$').match(res)
        assert len(res) > 3
        return res
    except:
        logging.error('Failed "git rev-parse --abbrev-ref HEAD" in "{}" (will ignore)'.format(source_dir), exc_info=True)
        return None


def find_sentry_release_id(repo_filename: str) -> Optional[str]:
    try:
        res = _call_sentry_cli("releases", "propose-version").strip()
        assert re.compile(r'^([0-9a-f]+)$').match(res)
        assert len(res) > 10
        logging.info('Sentry Release ID found: {}'.format(res))
        return res
    except:
        logging.info('Sentry Release ID not found using sentry-cli. Will use fallback using git.')

    # FALLBACK
    sha = _find_git_sha(repo_filename)
    # branch = _find_git_branch(repo_filename)
    if sha:
        res = sha
        logging.info('Sentry Release ID found: {}'.format(res))
        return res
    # if sha and branch:
    #     res = '{}-{}'.format(branch, sha)
    #     logging.info('Sentry Release ID found: {}'.format(res))
    #     return res
    else:
        logging.warning('Sentry Release ID NOT FOUND')
        return None