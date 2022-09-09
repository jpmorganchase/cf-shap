"""Human-readable strings."""

TIME_DURATION_UNITS = (
    ('year', 60 * 60 * 24 * 365),
    ('month', 60 * 60 * 24 * 30),
    ('week', 60 * 60 * 24 * 7),
    ('day', 60 * 60 * 24),
    ('hour', 60 * 60),
    ('min', 60),
    ('sec', 1),
)


def duration_to_long_string(
    seconds: float,
    nb_parts: int = 2,
    compact: bool = False,
    largest_unit: str = 'day',
    smallest_unit: str = 'sec',
    precision: int = 2,
) -> str:
    """Convert a number of seconds to a human-readable string.

    Args:
        seconds (float): Number of seconds.
        nb_parts (int, optional): Number of parts. Defaults to 2.
        compact (bool, optional): If True, the string will be more compact (use comma as separator). Defaults to False.
        largest_unit (str, optional): The largest unit to print. Defaults to 'day'.
        precision (int, optional): Decimal rounding precision. Defaults to 2.

    Returns:
        str: Duration in human-readable format.
    """

    assert seconds >= 0, "Duration must be greater than or equal to 0."
    assert nb_parts > 0, "Number of parts must be greater than 0."

    parts = []
    started = False
    started_writing = False

    for unit, div in TIME_DURATION_UNITS:

        # Continue until we reach the biggest unit
        if not unit == largest_unit and not started:
            continue

        started = True

        amount, seconds = divmod(seconds, div)

        if amount > 0:
            started_writing = True

        if started_writing:
            if nb_parts == 1 or smallest_unit == unit:
                excess = round(amount + seconds / div, precision)
                if excess > 0:
                    parts.append('{} {}{}'.format(excess, unit, "" if amount == 1 else "s"))
            else:
                if amount > 0:
                    parts.append('{} {}{}'.format(amount, unit, "" if amount == 1 else "s"))

            nb_parts -= 1

            if nb_parts == 0:
                break

        if smallest_unit == unit:
            break

    if len(parts) == 0:
        return 'now'
    elif len(parts) == 1:
        return parts[0]
    else:
        if not compact:
            return ', '.join(parts[:-1]) + ' and ' + parts[-1]
        else:
            return ', '.join(parts)


# Aliases
seconds_to_long_string = duration_to_long_string