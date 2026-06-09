import re
from typing import List

_BASH_NO_QUOTE_PATTERN = re.compile(r'[/:.A-Za-z0-9_-]+')


def need_bash_quote(arg: str) -> bool:
    """
    returns True if the argument must be quoted when used as a command line argument on a bash shell.
    """
    return len(arg) == 0 or not _BASH_NO_QUOTE_PATTERN.fullmatch(arg)


def bash_quote(arg: str) -> str:
    """
    Quotes the argument, for bash.
    This handles quote marks correctly for bash (which looks very ugly in bash).
    """
    if need_bash_quote(arg):
        return "'"+arg.replace("\'", "'\\''")+"'"
    else:
        return arg


def bash_cmd_array_to_str(cmd: List[str]) -> str:
    """
    Convert a command array, like you pass to Popen, to a string command that can safely be executed on a shell.
    This quotes the arguments needed.
    """
    return ' '.join([bash_quote(arg) for arg in cmd])
