import heapq
from functools import cache, reduce
from typing import Set
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

src_root = Path().resolve()
print(list(src_root.iterdir()))

default_img_path = src_root / "data" / "maze.png"


def load_photo(img_path=default_img_path):
    image = cv2.imread(str(img_path))
    assert image is not None
    return image


def viz(cv_img):
    plt.imshow(cv_img)
    plt.show()


bug_counter = 0


def get_unvisited_neighbours(element, matrix_shape, visited_set, frontier_set):
    # assume 2D matrix and that we are checking immediate neighbours - a 3*3 receptive field
    x_current, y_current = element
    x_end, y_end = matrix_shape

    # use +2 for upper bound to work around range behaviour at upper limits
    x_range = range(max(x_current - 1, 0), min(x_current + 2, x_end))
    y_range = range(max(y_current - 1, 0), min(y_current + 2, y_end))

    potential_elements_to_visit = [(x, y) for x in x_range for y in y_range]
    elements_to_visit = []

    for candidate_element in potential_elements_to_visit:
        if candidate_element == element:
            # do not readd the current node
            continue
        elif candidate_element not in visited_set:
            elements_to_visit.append(candidate_element)
        else:
            pass

    assert len(
        elements_to_visit) < 9, f"Too many neighbouring elements returned for a 2D pixel! {len(elements_to_visit)}"
    return elements_to_visit


def create_fitness_score(element, difficulty_map, output_values):
    # assume 2D matrix and that we are checking immediate neighbours - a 3*3 receptive field
    global bug_counter
    x_current, y_current = element
    x_end, y_end = output_values.shape

    x_range = range(max(x_current - 1, 0), min(x_current + 2, x_end))
    y_range = range(max(y_current - 1, 0), min(y_current + 2, y_end))

    penalty = get_pixel_penalty(difficulty_map, element)

    # take the minimum score of any neighbour then add the distance penalty
    # use filter to remove None i.e. unvisited neighbours. These can be ignored as
    # they will always be more expensive than earlier nodes.

    sqrt2 = np.sqrt(2)
    element_score = None
    for x in x_range:
        for y in y_range:
            if output_values[x, y] != np.inf:  # 0 is valid
                # make contextualised score
                if x != x_current and y != y_current:
                    candidate_score = output_values[x, y] + sqrt2 * penalty
                else:
                    candidate_score = output_values[x, y] + penalty

                # update output if this is better than the rest
                if element_score is None or element_score > candidate_score:
                    # update
                    element_score = candidate_score
                    if x == x_current and y == y_current:
                        print(f"found best value already in place! This should happen once. {bug_counter}")
                        if bug_counter > 0:
                            breakpoint
                        bug_counter = bug_counter + 1

    assert element_score != np.inf
    assert element_score is not None
    return element_score


def get_pixel_penalty(difficulty_map, element):
    x_current, y_current = element

    # distance penalty: lower on the prescribed path:
    # RGB -> red is idx 0
    is_red = bool(difficulty_map[x_current, y_current, 0])
    is_green = bool(difficulty_map[x_current, y_current, 1])
    is_blue = bool(difficulty_map[x_current, y_current, 2])
    if is_blue:
        penalty = 0
    elif is_red or is_green:
        penalty = 0.05
    else:
        penalty = 1
    return penalty


def generic_search_algorithm(seed_coord, visited_set: Set, difficulty_map, output_values):
    frontier_set = {0: seed_coord}

    # set up a min-heap to manage automatically sorting the queue
    queue = []
    heapq.heappush(queue, 0)

    matrix_shape = output_values.shape

    # tqdm progress bar up to the number of pixels
    total_pixels = reduce(lambda x, y: x * y, matrix_shape, 1)
    print(total_pixels)
    progress = trange(total_pixels)
    progress_iterator = iter(progress)

    while len(frontier_set) > 0:
        key = heapq.heappop(queue)
        element = frontier_set[key]
        del frontier_set[key]

        if element in visited_set:
            # already seen - skip
            continue

        current_element_fitness_score = create_fitness_score(element, difficulty_map, output_values)
        output_values[element] = current_element_fitness_score
        # Visit complete, come again soon!
        visited_set.add(element)

        unvisited_neighbours = get_unvisited_neighbours(element, matrix_shape, visited_set, frontier_set)

        new_keys = [extend_dict_non_clobbering(
                frontier_set,
                neighbour,
                current_element_fitness_score+taxicab_heuristic(neighbour, seed_coord)
            ) for neighbour in unvisited_neighbours
        ]

        for new_key in new_keys:
            heapq.heappush(queue, new_key)

        next(progress_iterator)
    progress.close()

    return output_values


def extend_dict_non_clobbering(dictionary, new_element, new_cost):
    if new_cost not in dictionary:
        dictionary[new_cost] = new_element
        return new_cost
    else:
        # increase cost by a tiny amount - less than the map would use!
        new_cost = new_cost + 0.000000001
        return extend_dict_non_clobbering(dictionary, new_element, new_cost)



@cache
def taxicab_heuristic(input_coords, goal_coords):
    in_x, in_y = input_coords
    goal_x, goal_y = goal_coords
    return abs(goal_x - in_x) + abs(goal_y - in_y)


def make_naive_graph_path(maze_path_img) -> np.ndarray:
    # start with the blue endzone
    endzone = maze_path_img[:, :, 2]
    # Step 1: Prepare a blank image
    score_matrix = np.full(endzone.shape[:2], dtype="float64", fill_value=np.inf)  # Creating a black canvas

    endzone_xs, endzone_ys = np.where(endzone)
    # any pixel in the endzone will do
    seed_pixel_coords = (endzone_xs[0], endzone_ys[0])
    print(f"seed pixel selected: {seed_pixel_coords}")
    score_matrix[seed_pixel_coords] = 0

    visited_set = set()

    scored_pixels = generic_search_algorithm(seed_coord=seed_pixel_coords, visited_set=visited_set,
                                             difficulty_map=maze_path_img, output_values=score_matrix)

    return scored_pixels


if __name__ == "__main__":
    annotated_route_path = src_root / "data" / "maze_path_manual.png"
    maze_path_manual = load_photo(annotated_route_path)
    print(f"src_root: {src_root}")
    print(f"manual annotation file location: {annotated_route_path}")
    viz(maze_path_manual)
    # mini_maze_path = cv2.resize(maze_path_manual, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

    gradient_map = make_naive_graph_path(maze_path_manual)
    # gradient_map = make_naive_graph_path(mini_maze_path)
    viz(gradient_map)

    outdir = src_root / "output"
    outdir.mkdir(exist_ok=True)
    outpath = outdir / "cost.csv"

    print(f"Writing output to: {outpath}")
    np.savetxt(outpath, gradient_map, delimiter='\t')

    cv2.imwrite(str(outdir / "cost.png"), gradient_map)
