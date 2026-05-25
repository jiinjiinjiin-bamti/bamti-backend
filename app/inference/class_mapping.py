from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceDetectionClass:
    variable_name: str
    display_name: str
    description: str
    raw_class_names: tuple[str, ...]
    threshold: float


raw_action_class_names = tuple(f"A{index}" for index in range(1, 17))

service_detection_classes = (
    ServiceDetectionClass(
        variable_name="normal_driving",
        display_name="정상 주행",
        description="A1 - 정상 주행",
        raw_class_names=("A1",),
        threshold=0.50,
    ),
    ServiceDetectionClass(
        variable_name="phone_use",
        display_name="휴대기기 조작",
        description="A5, A6, A7, A8, A9 - 휴대기기 조작",
        raw_class_names=("A5", "A6", "A7", "A8", "A9"),
        threshold=0.13,
    ),
    ServiceDetectionClass(
        variable_name="vehicle_device_operation",
        display_name="차량 장치 조작",
        description="A3, A4 - 차량 장치 조작",
        raw_class_names=("A3", "A4"),
        threshold=0.33,
    ),
    ServiceDetectionClass(
        variable_name="face_action",
        display_name="얼굴 행동",
        description="A2, A13, A14, A16 - 섭취, 흡연 등 얼굴 행동",
        raw_class_names=("A2", "A13", "A14", "A16"),
        threshold=0.16,
    ),
    ServiceDetectionClass(
        variable_name="distraction",
        display_name="주의 분산",
        description="A10, A11 - 대화 등 주의 분산",
        raw_class_names=("A10", "A11"),
        threshold=0.33,
    ),
    ServiceDetectionClass(
        variable_name="drowsiness",
        display_name="졸음",
        description="A12 - 졸음",
        raw_class_names=("A12",),
        threshold=0.65,
    ),
    ServiceDetectionClass(
        variable_name="rear_seat_interaction",
        display_name="뒷좌석 상호작용",
        description="A15 - 뒷좌석 상호작용",
        raw_class_names=("A15",),
        threshold=0.65,
    ),
)

driver4_raw_class_names = ("신체만짐", "주의분산", "핸드폰 조작", "핸들 조작")

driver4_service_detection_classes = (
    ServiceDetectionClass(
        variable_name="body_touching",
        display_name="신체 만짐",
        description="Driver4 model class: 신체만짐",
        raw_class_names=("신체만짐",),
        threshold=0.65,
    ),
    ServiceDetectionClass(
        variable_name="distraction",
        display_name="주의 분산",
        description="Driver4 model class: 주의분산",
        raw_class_names=("주의분산",),
        threshold=0.65,
    ),
    ServiceDetectionClass(
        variable_name="phone_operation",
        display_name="핸드폰 조작",
        description="Driver4 model class: 핸드폰 조작",
        raw_class_names=("핸드폰 조작",),
        threshold=0.65,
    ),
    ServiceDetectionClass(
        variable_name="steering_operation",
        display_name="핸들 조작",
        description="Driver4 model class: 핸들 조작",
        raw_class_names=("핸들 조작",),
        threshold=0.65,
    ),
)
