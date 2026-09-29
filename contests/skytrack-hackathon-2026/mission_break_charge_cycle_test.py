"""Short automatic landing -> charging -> takeoff test at CS1.

This test does not use SkyTrack Mission Break:

1. Take off to 20 m, fly 30 m out and return over CS1 at 20 m.
2. Land and disarm on the centre of CS1.
3. Wait on the ground for the battery to reach 99%.
4. If charging succeeds, take off to 20 m, fly a short verification leg,
   return to CS1 and land.
5. If charging is not observed within three minutes, remain safely on the
   ground and end the mission without trying to take off.
"""
from __future__ import annotations

from typing import Any, Iterator

from local_planner import boot_drone, brake, fly_to, land, takeoff


TAG = "[AUTO_CHARGE_TEST]"

# CS1: world ENU (x=east, y=north), valid landing radius 3 m.
CS1_EAST_M = 0.0
CS1_NORTH_M = 0.0

ALT_M = 20.0
SPEED_M_S = 5.0
PRE_CHARGE_NORTH_M = -30.0
POST_CHARGE_NORTH_M = -15.0

CHARGED_PERCENT = 99.0
CHARGE_TIMEOUT_S = 180.0
BATTERY_LOG_INTERVAL_S = 5.0


class WaitForGroundCharge:
    """Wait while disarmed, without publishing any flight setpoint."""

    name = "wait_for_ground_charge"

    def __init__(self, station_name: str) -> None:
        self.station_name = station_name
        self.ctx: Any = None
        self.started_at: float | None = None
        self.next_log_at: float | None = None
        self.charged = False
        self.timed_out = False

    def start(self, ctx: Any, params: Any = None) -> None:
        self.ctx = ctx
        self.started_at = ctx.world.now()
        self.next_log_at = self.started_at
        # LandSkill has completed and the vehicle is disarmed. Ensure no
        # offboard trajectory setpoints are sent while it sits on the pad.
        ctx.world.publish_enable_to_fly(False)
        ctx.world.log_info(
            f"{TAG} landed at {self.station_name}; waiting on the ground "
            f"for battery >= {CHARGED_PERCENT:.0f}%"
        )

    def cancel(self, ctx: Any, reason: str) -> None:
        # Keep result flags available to the mission after this skill ends.
        self.ctx = None

    @property
    def is_done(self) -> bool:
        if self.ctx is None or self.started_at is None:
            return False

        now = self.ctx.world.now()
        percent = self.ctx.senses.battery.percent

        if self.next_log_at is not None and now >= self.next_log_at:
            self.ctx.world.log_info(
                f"{TAG} charging at {self.station_name}: battery={percent}"
            )
            self.next_log_at = now + BATTERY_LOG_INTERVAL_S

        if percent is not None and percent >= CHARGED_PERCENT:
            self.charged = True
            self.ctx.world.log_info(
                f"{TAG} charge confirmed: battery={percent:.1f}%"
            )
            return True

        if now - self.started_at >= CHARGE_TIMEOUT_S:
            self.timed_out = True
            self.ctx.world.log_warn(
                f"{TAG} no automatic charge detected after "
                f"{CHARGE_TIMEOUT_S:.0f}s; battery={percent}"
            )
            return True

        return False


def automatic_charge_cycle_test(ctx: Any) -> Iterator[Any]:
    """Test whether an ordinary landing on CS1 starts charging."""
    ctx.world.log_info(f"{TAG} starting pre-charge flight")
    yield takeoff(alt_m=ALT_M)

    # Consume a little battery, then approach the pad horizontally at 20 m.
    yield fly_to(
        north=PRE_CHARGE_NORTH_M,
        east=CS1_EAST_M,
        alt_m=ALT_M,
        target_speed=SPEED_M_S,
        name="pre_charge_outbound",
    )
    yield fly_to(
        north=CS1_NORTH_M,
        east=CS1_EAST_M,
        alt_m=ALT_M,
        target_speed=SPEED_M_S,
        name="arrive_over_CS1_at_20m",
    )
    yield brake(name="settle_over_CS1")
    yield land(name="charge_land_CS1")

    charger = WaitForGroundCharge("CS1")
    yield charger

    if not charger.charged:
        ctx.world.log_warn(
            f"{TAG} test finished on the ground: ordinary land() did not "
            "produce a confirmed full charge"
        )
        return

    # Charging succeeded: prove that the generator can continue after land().
    ctx.world.log_info(f"{TAG} charged; taking off and continuing mission")
    yield takeoff(alt_m=ALT_M)
    yield fly_to(
        north=POST_CHARGE_NORTH_M,
        east=CS1_EAST_M,
        alt_m=ALT_M,
        target_speed=SPEED_M_S,
        name="post_charge_verification",
    )
    yield fly_to(
        north=CS1_NORTH_M,
        east=CS1_EAST_M,
        alt_m=ALT_M,
        target_speed=SPEED_M_S,
        name="final_return_CS1",
    )
    yield brake(name="final_settle_CS1")
    yield land(name="final_land_CS1")


automatic_charge_cycle_test.requires_senses = [
    "pose",
    "obstacle",
    "status",
    "battery",
]


def main() -> None:
    with boot_drone() as drone:
        drone.fly(automatic_charge_cycle_test)
        drone.run()


if __name__ == "__main__":
    main()
