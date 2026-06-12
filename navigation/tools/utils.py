import numpy as np

from paths import OUT_DIR




def combine_npz():
    data1 = np.load(OUT_DIR / "100b_300d_latitudes_1.npz")
    data2 = np.load(OUT_DIR / "100b_300d_latitudes_2.npz")
    data3 = np.load(OUT_DIR / "100b_300d_latitudes_3.npz")

    # Verify time vectors match
    assert np.array_equal(
        data1["times"],
        data2["times"]
    )

    assert np.array_equal(
        data1["times"],
        data3["times"]
    )

    combined = {
        "times": data1["times"],

        "altitude": np.concatenate(
            [
                data1["altitude"],
                data2["altitude"],
                data3["altitude"]
            ],
            axis=0
        ),

        "latitude": np.concatenate(
            [
                data1["latitude"],
                data2["latitude"],
                data3["latitude"]
            ],
            axis=0
        ),

        "longitude": np.concatenate(
            [
                data1["longitude"],
                data2["longitude"],
                data3["longitude"]
            ],
            axis=0
        )
    }

    np.savez_compressed(
        OUT_DIR / "300b_300d_combined.npz",
        **combined
    )

    print(
        "Combined shape:",
        combined["latitude"].shape
    )







if __name__ == "__main__":
    combine_npz()