import json
import argparse
import colorsys
from collections.abc import Iterable
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import transform

from PIL import Image, ImageDraw, ImageFont

from read_pixel_matrix import read_pixel_matrix


@dataclass(frozen=True)
class Seed:
    pixels: tuple[tuple[int, int], ...]

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        columns, rows = zip(*self.pixels)
        return min(columns), min(rows), max(columns), max(rows)


def saturation(pixel: tuple[int, ...]) -> float:
    red, green, blue = (channel / 255 for channel in pixel[:3])
    return colorsys.rgb_to_hsv(red, green, blue)[1] * 255


def find_seeds(
    pixel_matrix: list[list[tuple[int, ...]]],
    slice_height: int = 10,
    min_pixels: int = 5,
    saturation_threshold: float = 150,
) -> list[Seed]:
    """Find saturated 8-connected components within horizontal slices, bottom first."""
    if slice_height < 1 or min_pixels < 1:
        raise ValueError("slice_height and min_pixels must be positive")
    if not 0 <= saturation_threshold <= 255:
        raise ValueError("saturation_threshold must be between 0 and 255")

    height = len(pixel_matrix)
    width = len(pixel_matrix[0]) if height else 0
    seeds: list[Seed] = []

    for slice_end in range(height, 0, -slice_height):
        slice_start = max(0, slice_end - slice_height)
        candidates = {
            (column, row)
            for row in range(slice_start, slice_end)
            for column in range(width)
            if saturation(pixel_matrix[row][column]) >= saturation_threshold
        }

        while candidates:
            component = _pop_component(candidates)
            if len(component) >= min_pixels:
                seeds.append(Seed(tuple(component)))

    return seeds


def _pop_component(candidates: set[tuple[int, int]]) -> set[tuple[int, int]]:
    start = candidates.pop()
    component = {start}
    pending = [start]

    while pending:
        column, row = pending.pop()
        neighbors: Iterable[tuple[int, int]] = (
            (column + column_offset, row + row_offset)
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
            if column_offset or row_offset
        )
        for neighbor in neighbors:
            if neighbor in candidates:
                candidates.remove(neighbor)
                component.add(neighbor)
                pending.append(neighbor)

    return component


def draw_seeds(image_path: Path, seeds: list[Seed], output_path: Path, number_seeds: bool = False) -> None:
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGBA")

    overlay = Image.new("RGBA", annotated_image.size)
    overlay_draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(15)
    for i, seed in enumerate(seeds):
        for column, row in seed.pixels:
            overlay_draw.point((column, row), fill=(0, 255, 255, 255))
        overlay_draw.rectangle(seed.bounds, outline=(255, 0, 255, 255), width=1)
        if number_seeds == True:
         overlay_draw.text((seed.bounds[0], seed.bounds[1]), str(i), font=font, fill=(16, 255, 0, 255))

    Image.alpha_composite(annotated_image, overlay).save(output_path)


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(
        description="Find saturated 8-connected seed segments in horizontal image slices."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=project_root / "resources" / "ABCMO_294_detail.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "resources" / "ABCMO_294_detail_seeds.png",
    )
    parser.add_argument("--slice-height", "--sh", type=int, default=10, help="The height of the slices the image is divided into")
    parser.add_argument("--min-pixels", "--mp", type=int, default=5, help="The minimum amount of pixels required for the seed calculation")
    parser.add_argument("--saturation-threshold", "--st", type=float, default=150, help="The Threshold for saturation for the seed calculation")
    parser.add_argument("--min-slope", "--mns","--smn","--slope-min", type=float, default=0, help="The minimum amount of slope the transform will use")
    parser.add_argument("--max-slope", "--mxs","--smx","--slope-max", type=float, default=1, help="The maximum amount of slope the transform will use")
    parser.add_argument("--lines-amount", "--la", type=int, default=3, help="TEMPORARY UNTIL DYNAMIC IMPLEMENTATION! The amount of lines the transform will draw onto the seed")
    parser.add_argument("--number-seeds", "--ns","--num", action="store_true", help="Whether or not the seeds should be numbered in the image")
    parser.add_argument("--seed-index","--idx","--seed", type=int, default=None, help="The index of the seed you want to analyze. Defaults to the largest seed")
    parser.add_argument("--chart","--ch", action="store_true", help="Turn on the m/t chart")
    parser.add_argument("--only-all","--oa", action="store_true", help="Saves only the seed overlay for all found seeds")
    parser.add_argument("--clear-intersects","--ci", action="store_false", help="Stops clearing the Intersects list in transform.py before recalculating it")
    parser.add_argument("--print-intersects","--pi", type=int, help="Prints out Intersection points for the chosen seed")
    parser.add_argument("--no-lines","--nl", action="store_true", help="Disables line drawing")
    parser.add_argument("--length", type=int, default=None, help="Changes the length of the lines drawn")
    parser.add_argument("--dist","--distance","--dt", "--distance-threshold", type=float, default=20, help="The distance threshold of the clustering")
    parser.add_argument("--slope-threshold", type=float, default=0.1, help="The maximum slope difference allowed when clustering seeds")
    parser.add_argument("--all","--a", action="store_true", help="Draws all seeds")
    parser.add_argument("--show-required","--sr","--rqr", action="store_true", help="Shows the required intersects to troubleshoot Intersects out of Bounds errors")
    parser.add_argument("--cluster-one","--clo", action="store_true", help="Starts the clustering algorithm for one seed")
    parser.add_argument("--cluster-all","--cla", action="store_true", help="Starts the clustering algorithm for all seeds")
    parser.add_argument("--merge","--mrg", action="store_true", help="Merges the final lines when clustering")
    parser.add_argument("--simulate","--sim", action="store_true", help="Easy access to the simulated.png returned from the simulator")
    parser.add_argument("--legend","--leg", action="store_true", help="Shows the Legend for the clustering chart")
    return parser.parse_args()

def extract_pixel_coordinates(Seed):
    folder_name = "resources"
    file_name = "pixels.json"
    pixel_coordinates = list(Seed.pixels)
    script_dir = Path(__file__).parent.parent
    target_path = script_dir / folder_name / file_name
    with open(target_path, 'w') as f:
        json.dump(pixel_coordinates, f, indent = 4)
    

def find_largest_seed(seeds: list[Seed], retcase: int = 0):
    largest_seed = max(seeds, key=lambda seed: len(seed.pixels))
    largest_seed_index = seeds.index(max(seeds, key=lambda seed: len(seed.pixels)))
    print(f"Found largest seed at index {largest_seed_index}")
    if retcase <= 3 and retcase >= 0:
        match retcase:
            case 0:
                #print("Returning index of largest seed!", largest_seed_index)
                return(largest_seed_index)
            case 1:
                #print("Returning largest seed!", largest_index)
                return(largest_seed)
            case 2:
                #print("Returning index and largest seed!", largest_seed_index, largest_seed)
                return(largest_seed, largest_seed_index)
    else:
        return(largest_seed_index)

def save_specific_seed(seeds: list[Seed], n: int):
    if n < 0 or None:
        raise ValueError("Seed index must be positive or 0!")
    chosen_seed  = seeds[int(n)]
    chosen_path = arguments.output.with_name(f"{arguments.output.stem}_seed_{n}_{arguments.output.suffix}")
    line_path = arguments.output.with_name(f"{arguments.output.stem}_seed_{n}_transform_lines{arguments.output.suffix}")
    #line_pixel_path = arguments.output.with_name(f"{arguments.output.stem}_seed_{n}_line_diagram_{arguments.output.suffix}")
    #extract_pixel_coordinates(chosen_seed)
    draw_seeds(arguments.image, [chosen_seed], chosen_path)
    #transform.draw_pixel_lines(arguments.min_slope, arguments.max_slope, chosen_path, line_pixel_path)
    if not arguments.no_lines:
        transform.drawLines(chosen_path, arguments.lines_amount, line_path, arguments.min_slope, arguments.max_slope, arguments.clear_intersects, arguments.length, chosen_seed, arguments.show_required)
    print(f"Saved overlay for seed {n} to {chosen_path}")
    if arguments.print_intersects is not None:
        if int(arguments.print_intersects) >= 0:
            print("Intersection Points for this seed: ", transform.find_n_most_common_points(int(arguments.print_intersects), arguments.min_slope, arguments.max_slope, arguments.clear_intersects))
        else:
            raise ValueError("print-intersects must be positive or 0!")
    if arguments.chart:
        transform.draw_chart(arguments.min_slope, arguments.max_slope, f"Seed {n} m/t chart")

def find_nearby_seeds(seeds: list[Seed], target_seed: Seed, distance_threshold: float) -> list[Seed]:
    nearby_seeds = []
    target_bounds = target_seed.bounds
    target_center = ((target_bounds[0] + target_bounds[2]) / 2, (target_bounds[1] + target_bounds[3]) / 2)

    for seed in seeds:
        if seed == target_seed:
            continue
        seed_bounds = seed.bounds
        seed_center = ((seed_bounds[0] + seed_bounds[2]) / 2, (seed_bounds[1] + seed_bounds[3]) / 2)
        distance = ((target_center[0] - seed_center[0]) ** 2 + (target_center[1] - seed_center[1]) ** 2) ** 0.5
        if distance <= distance_threshold:
            nearby_seeds.append(seed)

    return nearby_seeds

def check_nearby_seeds_for_intersections(seeds: list[Seed], target_seed: Seed, distance_threshold: float):
    nearby_seeds = find_nearby_seeds(seeds, target_seed, distance_threshold)
    Intersecting_seeds = []
    for i in range(len(nearby_seeds)):
        for x in range(len(transform.SeedLines)):
            if(transform.line_intersects_seed(transform.SeedLines[0][x], nearby_seeds[i])):
                Intersecting_seeds.append(nearby_seeds[i])

    return Intersecting_seeds

def draw_intersecting_seeds(seeds: list[Seed], target_seed: Seed, distance_threshold: float): # unfinished
    cluster_path = arguments.output.with_name(f"{arguments.output.stem}_cluster_{arguments.output.suffix}")
    seeds_to_draw = [target_seed]
    seeds_to_draw.extend(check_nearby_seeds_for_intersections(seeds, target_seed, distance_threshold))
    draw_seeds(arguments.image, seeds_to_draw, cluster_path)
    #print("Length:", len(seeds_to_draw))
    for i in range(len(seeds_to_draw)):
        #extract_pixel_coordinates(seeds_to_draw[i])
        transform.drawLines(cluster_path, arguments.lines_amount, cluster_path, arguments.min_slope, arguments.max_slope, arguments.clear_intersects, arguments.length, seeds_to_draw[i], arguments.show_required)

def find_cluster(
    seeds: list[Seed],
    target_seed: Seed,
    lines_by_seed: dict[Seed, list],
    distance_threshold: float,
    slope_threshold: float,
    intercept_threshold: float,
) -> list[Seed]:
    cluster = [target_seed]
    visited = {target_seed}
    queue = deque([target_seed])

    while queue:
        current_seed = queue.popleft()
        current_lines = lines_by_seed.get(current_seed, [])

        if not current_lines:
            continue

        current_slope, current_intercept = transform.average_line(current_lines)

        for candidate in find_nearby_seeds(
            seeds, current_seed, distance_threshold
        ):
            if candidate in visited:
                continue

            candidate_lines = lines_by_seed.get(candidate, [])
            if not candidate_lines:
                print("Rejected: no lines")
                continue

            candidate_slope, candidate_intercept = transform.average_line(
                candidate_lines
            )

            slope_difference = abs(candidate_slope - current_slope)
            intercept_difference = abs(
                candidate_intercept - current_intercept
            )

            if (
                slope_difference <= slope_threshold
                and intercept_difference <= intercept_threshold
            ):
                visited.add(candidate)
                cluster.append(candidate)
                queue.append(candidate)

    return cluster

def draw_cluster(seeds: list[Seed], target_seed: Seed, distance_threshold: float):
    lines_by_seed = {}

    for seed in seeds:
        lines = transform.getLines(
            arguments.min_slope,
            arguments.max_slope,
            seed,
            arguments.lines_amount,
            True,
            arguments.length or transform.dynamicLength(seed, 0.5),
            arguments.show_required
        )
        if lines:
            lines_by_seed[seed] = lines

    cluster = find_cluster(
        seeds,
        target_seed,
        lines_by_seed,
        distance_threshold=distance_threshold,
        slope_threshold=arguments.slope_threshold,
        intercept_threshold=20,
    )
    cluster_path = arguments.output.with_name(
        f"{arguments.output.stem}_cluster_{arguments.output.suffix}"
    )
    draw_seeds(arguments.image, cluster, cluster_path)

    for seed in cluster:
        #extract_pixel_coordinates(seed)
        transform.drawLines(
            cluster_path, arguments.lines_amount, cluster_path,
            arguments.min_slope, arguments.max_slope,
            arguments.clear_intersects, arguments.length, seed, arguments.show_required
        )

    unified_line = transform.line_for_cluster(cluster, lines_by_seed)
    if unified_line is not None:
        transform.draw_line(cluster_path, cluster_path, unified_line, color="blue")


def draw_clusters(seeds: list[Seed], distance_threshold: float) -> list:
    lines_by_seed = {}

    for seed in seeds:
        lines = transform.getLines(
            arguments.min_slope,
            arguments.max_slope,
            seed,
            arguments.lines_amount,
            True,
            arguments.length or transform.dynamicLength(seed, 0.5),
            arguments.show_required,
        )
        if lines:
            lines_by_seed[seed] = lines

    unassigned = set(seeds)
    ordered_seeds = sorted(
        seeds,
        key=lambda seed: len(seed.pixels),
        reverse=True,
    )

    all_path = arguments.output.with_name(
        f"{arguments.output.stem}_clusters{arguments.output.suffix}"
    )
    draw_seeds(arguments.image, seeds, all_path)

    cluster_lines = []
    cluster_line_data = []
    cluster_number = 0
    for target_seed in ordered_seeds:
        if target_seed not in unassigned:
            continue

        if target_seed not in lines_by_seed:
            unassigned.remove(target_seed)
            continue

        cluster = find_cluster(
            seeds,
            target_seed,
            lines_by_seed,
            distance_threshold=distance_threshold,
            slope_threshold=arguments.slope_threshold,
            intercept_threshold=20,
        )
        cluster = [seed for seed in cluster if seed in lines_by_seed]
        if not cluster:
            continue

        unassigned.difference_update(cluster)
        for seed in cluster:
            transform.drawLines(
                all_path,
                arguments.lines_amount,
                all_path,
                arguments.min_slope,
                arguments.max_slope,
                True,
                arguments.length,
                seed,
                arguments.show_required,
            )

        unified_line = transform.line_for_cluster(cluster, lines_by_seed)
        if unified_line is not None:
            transform.draw_line(
                all_path,
                all_path,
                unified_line,
                color="blue",
            )
            cluster_lines.append(unified_line)
            slope, intercept = transform.line_parameters(unified_line)
            cluster_line_data.append(
                {
                    "cluster": cluster_number,
                    "target_seed": seeds.index(target_seed),
                    "seed_indices": [seeds.index(seed) for seed in cluster],
                    "slope": slope,
                    "intercept": intercept,
                    "start": [float(unified_line.p1.x), float(unified_line.p1.y)],
                    "end": [float(unified_line.p2.x), float(unified_line.p2.y)],
                }
            )

        print(
            f"Saved cluster {cluster_number} from seed "
            f"{seeds.index(target_seed)} ({len(cluster)} seeds) to "
            f"{all_path}"
        )
        cluster_number += 1

    lines_path = all_path.with_suffix(".json")
    with open(lines_path, "w") as file:
        json.dump(cluster_line_data, file, indent=2)

    print(f"Saved {len(cluster_lines)} unified cluster lines to {lines_path}")
    return cluster_lines

def draw_all_seeds(seeds: list[Seed]):
    all_path = arguments.output.with_name(f"{arguments.output.stem}_height_{arguments.slice_height}_all_seeds_{arguments.output.suffix}")
    draw_seeds(arguments.image, seeds, all_path)
    if not arguments.no_lines:
        for x in range(len(seeds)):
            print(f"Drawing seed {x}")
            transform.drawLines(all_path, arguments.lines_amount, all_path, arguments.min_slope, arguments.max_slope, arguments.clear_intersects, arguments.length, seeds[x], arguments.show_required)
            print(f"Finished drawing seed {x}")
        print(f"Saved overlay for all seeds to {all_path}")

if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.simulate:
        arguments.image = Path("bubbles/resources/simulated.png")
        arguments.output = Path("bubbles/resources/simulated_detail.png")
    matrix = read_pixel_matrix(arguments.image)
    found_seeds = find_seeds(
        matrix,
        slice_height=arguments.slice_height,
        min_pixels=arguments.min_pixels,
        saturation_threshold=arguments.saturation_threshold,
    )
    if arguments.seed_index is None:
        if not arguments.only_all:
            arguments.seed_index = find_largest_seed(found_seeds, 0)
    draw_seeds(arguments.image, found_seeds, arguments.output, arguments.number_seeds)
    print(f"Found {len(found_seeds)} seeds. Saved overlay to {arguments.output}.")
    if not arguments.only_all:
        if not arguments.all:
            save_specific_seed(found_seeds, arguments.seed_index)
            if arguments.cluster_one:
                draw_cluster(found_seeds, found_seeds[arguments.seed_index], arguments.dist)
            elif arguments.cluster_all:
                draw_clusters(found_seeds, arguments.dist)
                transform.plot_lines(f"{arguments.output.stem}_clusters.json", arguments.merge, arguments.legend)
        else:
            draw_all_seeds(found_seeds)
        #print(transform.getLines(arguments.min_slope, arguments.max_slope, found_seeds[arguments.seed_index],3,arguments.clear_intersects,arguments.length))