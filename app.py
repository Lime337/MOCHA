import streamlit as st
import numpy as np


st.set_page_config(
    page_title="MOCHA Demo",
    page_icon="🧠",
    layout="wide"
)

st.title("MOCHA")
st.subheader("Modular Continual Hyperdimensional Adaptation")

st.write(
    "A lightweight visual demonstration of the MOCHA framework across "
    "security, vision, and wearable sensing."
)

st.caption(
    "Demo values shown below are illustrative and are not experimental results."
)

st.markdown("---")


if "application" not in st.session_state:
    st.session_state.application = "HERMES"

if "state" not in st.session_state:
    st.session_state.state = "baseline"


st.markdown("## Select Application")

c1, c2, c3 = st.columns(3)

with c1:
    if st.button(
        "🛡️ HERMES\n\nAndroid Malware Detection",
        use_container_width=True
    ):
        st.session_state.application = "HERMES"
        st.session_state.state = "baseline"
        st.rerun()

with c2:
    if st.button(
        "🧍 POSEiDON\n\nHuman Pose Classification",
        use_container_width=True
    ):
        st.session_state.application = "POSEiDON"
        st.session_state.state = "baseline"
        st.rerun()

with c3:
    if st.button(
        "⌚ THEMIS\n\nWearable Activity Recognition",
        use_container_width=True
    ):
        st.session_state.application = "THEMIS"
        st.session_state.state = "baseline"
        st.rerun()


application = st.session_state.application


APP_INFO = {
    "HERMES": {
        "input": "APK / Opcode Data",
        "encoding": "Opcode + Method Encoding",
        "prediction": "Benign / Malware",
        "description": (
            "HERMES converts Android opcode behavior into hypervectors "
            "for lightweight malware classification."
        ),
        "baseline": 92.0,
        "corrupted": 82.0,
    },

    "POSEiDON": {
        "input": "Pose Landmarks",
        "encoding": "Spatial + Angle Encoding",
        "prediction": "Pose Class",
        "description": (
            "POSEiDON converts body landmarks and joint angles into "
            "hypervectors for human pose classification."
        ),
        "baseline": 96.0,
        "corrupted": 86.0,
    },

    "THEMIS": {
        "input": "Sensor Window",
        "encoding": "Temporal Sensor Encoding",
        "prediction": "Activity Class",
        "description": (
            "THEMIS converts wearable sensor windows into hypervectors "
            "and supports personalization and reversible updates."
        ),
        "baseline": 93.0,
        "corrupted": 84.0,
    }
}

info = APP_INFO[application]

st.markdown(f"### {application}")
st.info(info["description"])


st.markdown("## MOCHA Processing Pipeline")

p1, p2, p3, p4, p5, p6, p7 = st.columns(
    [2, 0.5, 2, 0.5, 2, 0.5, 2]
)

with p1:
    st.markdown("### Input")
    st.write(info["input"])

with p2:
    st.markdown("### →")

with p3:
    st.markdown("### Encoder")
    st.write(info["encoding"])

with p4:
    st.markdown("### →")

with p5:
    st.markdown("### Hypervector")
    st.write("Bipolar representation")
    st.code("h ∈ {-1, +1}ᴰ")

with p6:
    st.markdown("### →")

with p7:
    st.markdown("### Prediction")
    st.write(info["prediction"])

st.markdown("---")


st.markdown("## Reversible Adaptation")

st.write(
    "MOCHA retains additive prototype information so that incorrect "
    "contributions can later be algebraically removed."
)

b1, b2, b3 = st.columns(3)

with b1:
    if st.button(
        "⚠️ Inject Bad Data",
        use_container_width=True
    ):
        st.session_state.state = "corrupted"
        st.rerun()

with b2:
    if st.button(
        "↩️ Unlearn",
        use_container_width=True
    ):
        st.session_state.state = "restored"
        st.rerun()

with b3:
    if st.button(
        "🔄 Reset",
        use_container_width=True
    ):
        st.session_state.state = "baseline"
        st.rerun()


state = st.session_state.state

if state == "baseline":

    accuracy = info["baseline"]

    state_name = "Baseline"

    state_message = (
        "Original model state."
    )

    state_icon = "✅"

elif state == "corrupted":

    accuracy = info["corrupted"]

    state_name = "Corrupted"

    state_message = (
        "Incorrect contributions have been added "
        "to the class prototype."
    )

    state_icon = "⚠️"

else:

    accuracy = info["baseline"]

    state_name = "Restored"

    state_message = (
        "MOCHA algebraically removed the incorrect "
        "contributions and restored the original state."
    )

    state_icon = "✅"


m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Demo Model State",
        state_name
    )

with m2:

    st.metric(
        "Illustrative Accuracy",
        f"{accuracy:.1f}%"
    )

with m3:

    if state == "corrupted":

        delta = (
            info["corrupted"]
            - info["baseline"]
        )

    elif state == "restored":

        delta = (
            info["baseline"]
            - info["corrupted"]
        )

    else:

        delta = 0

    st.metric(
        "Change",
        f"{delta:+.1f} percentage points"
    )


st.progress(accuracy / 100)

if state == "corrupted":

    st.error(
        f"{state_icon} {state_message}"
    )

else:

    st.success(
        f"{state_icon} {state_message}"
    )


st.markdown("### Adaptation Flow")

f1, f2, f3, f4, f5 = st.columns(
    [2, 0.5, 2, 0.5, 2]
)

with f1:

    if state == "baseline":

        st.success(
            "1. Baseline Model"
        )

    else:

        st.write(
            "1. Baseline Model"
        )

with f2:

    st.markdown("### →")

with f3:

    if state == "corrupted":

        st.error(
            "2. Corrupted Model"
        )

    else:

        st.write(
            "2. Corrupted Model"
        )

with f4:

    st.markdown("### →")

with f5:

    if state == "restored":

        st.success(
            "3. Restored Model"
        )

    else:

        st.write(
            "3. Restored Model"
        )


st.markdown("---")


st.markdown("## Prototype Visualization")

st.write(
    "Each square represents one displayed coordinate of a much larger "
    "high-dimensional class prototype."
)


rng = np.random.default_rng(42)

baseline_hv = rng.choice(
    [-1, 1],
    size=32
)


corrupted_hv = baseline_hv.copy()

flip_indices = [
    3,
    7,
    12,
    18,
    22,
    27
]

corrupted_hv[flip_indices] *= -1


def render_hv(vector, changed_indices=None):

    boxes = []

    for i, value in enumerate(vector):


        if (
            changed_indices is not None
            and i in changed_indices
        ):

            border = "3px solid #ff4b4b"

        else:

            border = "1px solid #666"


        if value == 1:

            background = "#4da3ff"

        else:

            background = "#d9d9d9"

        box = (
            '<span style="'
            'display:inline-block;'
            'width:28px;'
            'height:28px;'
            'margin:3px;'
            f'background:{background};'
            f'border:{border};'
            'border-radius:4px;'
            'box-sizing:border-box;'
            '"></span>'
        )

        boxes.append(box)

    html = (
        '<div style="'
        'display:flex;'
        'flex-wrap:wrap;'
        'align-items:center;'
        'width:100%;'
        'padding:14px;'
        'background:#161b22;'
        'border:1px solid #30363d;'
        'border-radius:8px;'
        'box-sizing:border-box;'
        '">'
        + "".join(boxes)
        + "</div>"
    )

    return html


if state == "baseline":

    st.markdown(
        render_hv(
            baseline_hv
        ),
        unsafe_allow_html=True
    )

    st.caption(
        "Original class prototype."
    )


elif state == "corrupted":

    st.markdown(
        render_hv(
            corrupted_hv,
            changed_indices=flip_indices
        ),
        unsafe_allow_html=True
    )

    st.caption(
        "Red outlines identify coordinates altered "
        "by the simulated bad update."
    )


else:

    st.markdown(
        render_hv(
            baseline_hv
        ),
        unsafe_allow_html=True
    )

    st.caption(
        "The original prototype has been restored "
        "after unlearning."
    )


st.markdown("### MOCHA Update Operation")

if state == "baseline":

    st.code(
        "Original state:     P_c"
    )

elif state == "corrupted":

    st.code(
        "Bad update:         P_c ← P_c + h_bad"
    )

else:

    st.code(
        "Unlearning:         P_c ← P_c - h_bad"
    )


st.markdown("### Hypervector Legend")

l1, l2, l3 = st.columns(3)


with l1:

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<span style="'
        'display:inline-block;'
        'width:22px;'
        'height:22px;'
        'background:#4da3ff;'
        'border:1px solid #666;'
        'border-radius:4px;'
        '"></span>'
        '<span><b>+1</b></span>'
        '</div>',
        unsafe_allow_html=True
    )


with l2:

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<span style="'
        'display:inline-block;'
        'width:22px;'
        'height:22px;'
        'background:#d9d9d9;'
        'border:1px solid #666;'
        'border-radius:4px;'
        '"></span>'
        '<span><b>-1</b></span>'
        '</div>',
        unsafe_allow_html=True
    )


with l3:

    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<span style="'
        'display:inline-block;'
        'width:22px;'
        'height:22px;'
        'background:#4da3ff;'
        'border:3px solid #ff4b4b;'
        'border-radius:4px;'
        'box-sizing:border-box;'
        '"></span>'
        '<span><b>Changed by bad update</b></span>'
        '</div>',
        unsafe_allow_html=True
    )


st.markdown("---")

st.caption(
    "MOCHA Demo | HERMES • POSEiDON • THEMIS"
)
