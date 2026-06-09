import copy
import datetime
import re
from collections import deque
from typing import Optional, Any


def normalize_depenv(deployment_environment: str) -> str:
    if deployment_environment.lower() == 'stable':
        return 'production'
    if deployment_environment.lower() == 'prod':
        return 'production'
    if deployment_environment.lower() == 'production':
        return 'production'
    if deployment_environment.lower() == 'staging':
        return 'staging'
    if deployment_environment.lower() == 'dev':
        return 'staging'
    if deployment_environment.lower() == 'development':
        return 'staging'
    return deployment_environment


def datetime_now(zulu=False, no_milliseconds=True) -> datetime.datetime:
    res = datetime.datetime.now(datetime.timezone.utc)
    if not zulu:
        res = res.astimezone(datetime.timezone(datetime.timedelta(hours=0), name="UTC"))
    if no_milliseconds:
        res = res.replace(microsecond=0)
    assert res.tzinfo is not None  # enforce timezone aware (non-naive) datetime
    return res


def any_to_opt_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if value is None:
        return default
    if value == 1:
        return True
    if value == 0:
        return False
    if str(value).lower() in ['true', 't', '1', 'yes']:
        return True
    if str(value).lower() in ['false', 'f', '0', 'no']:
        return False
    return default


def any_to_bool(value: Any, default: Optional[bool] = None) -> bool:
    res = any_to_opt_bool(value, default)
    if res is not None:
        return res
    if default is None:
        raise ValueError('Could not convert "{}" to boolean value'.format(value))
    return default


def duration_string_to_seconds(duration: str) -> Optional[int]:
    if not duration:
        return None
    match = re.match(r'^([0-9.]+) *([a-zA-Z]+)$', duration)
    if not match:
        return None

    num = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
    unit = match.group(2).lower()
    if unit == 'h' or unit == 'hour' or unit == 'hours':
        return int(num * 60 * 60)
    elif unit == 'm' or unit == 'min' or unit == 'mins' or unit == 'minute' or unit == 'minutes':
        return int(num * 60)
    elif unit == 's' or unit == 'sec' or unit == 'secs' or unit == 'second' or unit == 'seconds':
        return int(num)
    elif unit == 'day' or unit == 'days':
        return int(num * 60 * 60 * 24)
    elif unit == 'week' or unit == 'weeks':
        return int(num * 60 * 60 * 24 * 7)
    else:
        return None
    # Impossible to support because not fixed length: month, year, ...


def deep_update_dict(target: dict, extra: dict) -> dict:
    """
    Deep update by adding all keys (nested) of extra to target.
    Nested arrays are copied from extra to target with copy.deepcopy(), so no deep_update is done on them!

    Notes:
       - this deeply modifies target, so make sure you first copy.deepcopy it if needed!
       - you can't remove keys from target with this, only add them
       - type conflicts between values of target and extra are handled by extra overriding target

    :param target: the dict that should be updated deeply
    :param extra: extra keys to be added (nested)
    :return: target
    """

    # deque to avoid recursion
    jobs = deque()
    jobs.append( (target, extra) )

    while len(jobs) > 0:
        job_target, job_extra = jobs.popleft()

        for key,extra_value in job_extra.items():
            if isinstance(extra_value, dict):
                if key in job_target and isinstance(job_target[key], dict):
                    jobs.append( (job_target[key], extra_value) )
                else:
                    job_target[key] = copy.deepcopy(extra_value)
            else:
                job_target[key] = extra_value

    return target


def strip_null_from_json_dict(json_dict: dict, *,
                              strip_empty_dict: bool = False,
                              strip_empty_list: bool = False,
                              process_lists: bool = False,
                              ):
    """
    Strip null values from dicts, also stripping empty dicts and list values if requested.
    Doesn't look inside lists unless process_lists=True, so doesn't:
       - Strip dicts inside lists.
       - Remove empty dicts inside lists.

    (recursive-less implementation, which adds a bit of complexity.)

    :param json_dict: The original json dict. Will not be modified.
    :param strip_empty_dict: remove empty dicts
    :param strip_empty_list: remove empty lists
    :param process_lists: modify nested lists
    :return:
    """
    res = copy.deepcopy(json_dict)

    todo = deque()
    todo.append((res, []))

    while len(todo) > 0:
        current, parents = todo.popleft()
        current_is_dict = isinstance(current, dict)

        if current_is_dict:
            empty = []
            for key, value in current.items():
                if value is None:
                    empty.append(key)
                elif isinstance(value, dict):
                    if len(value) > 0:
                        todo.append((value, parents + [current]))
                    if len(value) == 0 and strip_empty_dict:
                        empty.append(key)
                elif isinstance(value, list):
                    if len(value) > 0 and process_lists:
                        todo.append((value, parents + [current]))
                    if len(value) == 0 and strip_empty_list:
                        empty.append(key)
            for key in empty:
                del current[key]
        else:
            assert isinstance(current, list)
            empty_idx = []
            for i, value in enumerate(current):
                if value is None:
                    empty_idx.append(i)
                elif isinstance(value, dict):
                    if len(value) > 0:
                        todo.append((value, parents + [current]))
                    if len(value) == 0 and strip_empty_dict:
                        empty_idx.append(i)
                elif isinstance(value, list):
                    if len(value) > 0 and process_lists:
                        todo.append((value, parents + [current]))
                    if len(value) == 0 and strip_empty_list:
                        empty_idx.append(i)
            for i in reversed(empty_idx):
                del current[i]

        if len(current) == 0 and parents and (strip_empty_dict if current_is_dict else strip_empty_list):
            parent_parents = list(parents)
            parent = parent_parents.pop()
            todo.append((parent, parent_parents))
    return res
