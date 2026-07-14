from __future__ import annotations

from typing import Any


def build_region_grid(width: int, height: int, rows: int, columns: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for row in range(rows):
        y0 = (row * height) // rows
        y1 = ((row + 1) * height) // rows
        for column in range(columns):
            x0 = (column * width) // columns
            x1 = ((column + 1) * width) // columns
            regions.append(
                {
                    "region_id": f"region-r{row:02d}-c{column:02d}",
                    "row": row,
                    "column": column,
                    "normalized_bounds": {
                        "x": round(x0 / width, 6),
                        "y": round(y0 / height, 6),
                        "width": round((x1 - x0) / width, 6),
                        "height": round((y1 - y0) / height, 6),
                    },
                    "pixel_bounds": {
                        "x": int(x0),
                        "y": int(y0),
                        "width": int(max(0, x1 - x0)),
                        "height": int(max(0, y1 - y0)),
                    },
                }
            )
    return regions


def region_slice(region: dict[str, Any]) -> tuple[slice, slice]:
    bounds = region["pixel_bounds"]
    y0 = bounds["y"]
    x0 = bounds["x"]
    return (
        slice(y0, y0 + bounds["height"]),
        slice(x0, x0 + bounds["width"]),
    )
