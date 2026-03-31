import argparse
from pathlib import Path


def build_bddl(language: str, container_name: str, container_type: str) -> str:
    return f"""(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {language})
  (:regions
    (bin_region (:target floor) (:ranges ((-0.025 0.242 0.025 0.278))))
    (target_object_region (:target floor) (:ranges ((-0.16 -0.28 -0.08 -0.20))))
    (obstacle_region (:target floor) (:ranges ((-0.02 -0.03 0.02 0.03))))
    (contain_region (:target {container_name}))
  )
  (:fixtures
    floor - floor
    obstacle_1 - white_storage_box
  )
  (:objects
    milk_1 - milk
    {container_name} - {container_type}
  )
  (:obj_of_interest milk_1 {container_name})
  (:init
    (On milk_1 floor_target_object_region)
    (On {container_name} floor_bin_region)
    (On obstacle_1 floor_obstacle_region)
  )
  (:goal (And (In milk_1 {container_name}_contain_region)))
)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data_collection_outputs/bddl/dual_strategy_milk_obstacle.bddl"),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pick up and move the object to the target area",
    )
    parser.add_argument("--container-name", type=str, default="basket_1")
    parser.add_argument("--container-type", type=str, default="basket")
    args = parser.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        build_bddl(args.language, args.container_name, args.container_type),
        encoding="utf-8",
    )
    print(f"Wrote BDDL: {args.output_file}")


if __name__ == "__main__":
    main()
