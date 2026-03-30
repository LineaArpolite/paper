import argparse
from pathlib import Path


def build_bddl(language: str, container_name: str, container_type: str) -> str:
    return f"""(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {language})
  (:regions
    (bin_region (:target floor) (:ranges ((-0.02 0.245 0.02 0.275))))
    (target_object_region (:target floor) (:ranges ((-0.155 -0.275 -0.085 -0.205))))
    (obstacle_region (:target floor) (:ranges ((-0.02 -0.03 0.02 0.03))))
    (other_object_region_0 (:target floor) (:ranges ((0.025 -0.125 0.075 -0.075))))
    (other_object_region_1 (:target floor) (:ranges ((-0.175 0.035 -0.125 0.085))))
    (other_object_region_2 (:target floor) (:ranges ((0.075 -0.225 0.125 -0.175))))
    (other_object_region_3 (:target floor) (:ranges ((0.125 0.005 0.175 0.055))))
    (other_object_region_4 (:target floor) (:ranges ((-0.225 -0.105 -0.175 -0.055))))
    (contain_region (:target {container_name}))
  )
  (:fixtures
    floor - floor
    obstacle_1 - white_storage_box
  )
  (:objects
    milk_1 - milk
    {container_name} - {container_type}
    cream_cheese_1 - cream_cheese
    tomato_sauce_1 - tomato_sauce
    butter_1 - butter
    orange_juice_1 - orange_juice
    chocolate_pudding_1 - chocolate_pudding
  )
  (:obj_of_interest milk_1 {container_name})
  (:init
    (On milk_1 floor_target_object_region)
    (On cream_cheese_1 floor_other_object_region_0)
    (On tomato_sauce_1 floor_other_object_region_1)
    (On butter_1 floor_other_object_region_2)
    (On orange_juice_1 floor_other_object_region_3)
    (On chocolate_pudding_1 floor_other_object_region_4)
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
