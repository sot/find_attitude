# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from logging import CRITICAL

from django.shortcuts import render
from kadi import __version__

from find_attitude.find_attitude import (
    find_attitude_solutions,
    get_stars_from_text,
    logger,
)

# Only emit critical messages
logger.level = CRITICAL
# FIXME: this looks like a bug in the original code. Either get rid of the loop or
# set levels for all handlers along with the logger.
for _hdlr in logger.handlers:
    logger.setLevel(CRITICAL)


def index(request):
    if request.POST:
        context = find_solutions_and_get_context(request)
    else:
        context = {}

    context["kadi_version"] = __version__
    context["distance_tolerance"] = 2.5

    if context.get("solutions"):
        context["subtitle"] = ": Solution"
    elif context.get("error_message"):
        context["subtitle"] = ": Error"

    return render(request, "find_attitude/index.html", context)


def find_solutions_and_get_context(request):
    stars_text = request.POST.get("stars_text", "")
    context = {"stars_text": stars_text}

    # First parse the star text input
    try:
        stars = get_stars_from_text(stars_text)
    except Exception as err:
        context["error_message"] = "{}\nDoes it look like one of the examples?".format(
            err
        )
        return context

    # Try to find solutions
    tolerance = float(request.POST.get("distance_tolerance", "2.5"))
    try:
        solutions = find_attitude_solutions(stars, tolerance=tolerance)
    except Exception:
        import traceback

        context["error_message"] = traceback.format_exc()
        return context

    # No solutions, bummer.
    if not solutions:
        context["error_message"] = (
            "No matching solutions found.\n"
            "Try increasing distance and/or magnitude tolerance."
        )
        return context

    context["solutions"] = []
    for solution in solutions:
        summary_lines = solution["summary"].pformat(max_width=-1, max_lines=-1)
        sol = {
            "att_fit": solution["att_fit"],
            "summary": os.linesep.join(summary_lines),
        }
        context["solutions"].append(sol)

    return context
