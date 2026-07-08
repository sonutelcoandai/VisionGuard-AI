from app.core.config import ZONES


def point_in_poly(pt, poly):

    x, y = pt

    inside = False

    n = len(poly)

    for i in range(n):

        x1, y1 = poly[i]

        x2, y2 = poly[(i + 1) % n]

        cond = (
            ((y1 > y) != (y2 > y))
            and
            (
                x <
                (
                    (x2 - x1)
                    * (y - y1)
                    / (y2 - y1 + 1e-9)
                    + x1
                )
            )
        )

        if cond:
            inside = not inside

    return inside


def bbox_center(bb):

    x1, y1, x2, y2 = bb

    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )


def crosses_line(
    p_prev,
    p_now,
    line
):

    x3, y3, x4, y4 = line

    x1, y1 = p_prev

    x2, y2 = p_now

    def ccw(A, B, C):

        return (
            (C[1] - A[1])
            * (B[0] - A[0])
            >
            (B[1] - A[1])
            * (C[0] - A[0])
        )

    A = (x1, y1)
    B = (x2, y2)
    C = (x3, y3)
    D = (x4, y4)

    return (
        ccw(A, C, D)
        !=
        ccw(B, C, D)
    ) and (
        ccw(A, B, C)
        !=
        ccw(A, B, D)
    )


def zone_of_point(pt):

    for zone_name, poly in ZONES.items():

        try:

            if point_in_poly(
                pt,
                poly
            ):

                if zone_name == "cloth":
                    return "Cloth Section"

                return (
                    zone_name.title()
                    if zone_name.islower()
                    else zone_name
                )

        except Exception:
            continue

    return "Cloth Section"