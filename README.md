# FaceRecognition

A lightweight face recognition system built with **OpenCV**, **InsightFace**, **NumPy**, and **SQLite**.
It detects faces in video frames, compares them against previously stored face embeddings, and can automatically register new people after seeing them consistently across multiple frames.

## Features

* **Face detection and embedding extraction** using `insightface`
* **Persistent face database** stored in SQLite
* **Cosine similarity matching** against saved identities
* **Automatic unknown face tracking**
* **Automatic new person registration** after repeated sightings
* **Embedding updates over time** for recognized people to improve robustness

---

## How It Works

The `FaceRecognition` class processes video frames and performs the following steps:

1. **Detect faces** in the frame using `InsightFace`.
2. **Extract face embeddings** for each detected face.
3. **Compare each embedding** against stored embeddings in the SQLite database.
4. Based on similarity thresholds:

   * If the face matches a known person strongly enough, it is labeled with that person’s name.
   * If the face does not match any known person and remains consistently visible for several frames, a **new identity** is automatically created.
   * If the confidence is ambiguous, it is labeled as `"..."`.

---

# Project Structure

```bash
project/
│── face_recognition.py   # Main FaceRecognition class
│── faces.db              # SQLite database storing persons and embeddings
│── README.md             # Documentation
```

---

# Requirements

Install the required dependencies before running the project.

```bash
pip install opencv-python numpy insightface
```

SQLite is included with Python’s standard library, so no extra installation is required for it.

Depending on your environment, `insightface` may also require additional runtime dependencies such as `onnxruntime` or GPU support packages.

---

# Class Overview

## `FaceRecognition`

Main class responsible for:

* Loading and saving face embeddings from/to SQLite
* Running face analysis on frames
* Matching detected faces against known people
* Tracking unknown faces across frames
* Automatically registering new persons

---

# Initialization

```python
fr = FaceRecognition(db_file="faces.db")
```

## Parameters

### `db_file`

* **Type:** `str`
* **Default:** `"faces.db"`
* Path to the SQLite database file.

---

# Internal Configuration

These values control the recognition and auto-registration behavior:

| Attribute                 | Default | Description                                                                                              |
| ------------------------- | ------: | -------------------------------------------------------------------------------------------------------- |
| `MATCH_THRESHOLD`         |   `0.6` | Minimum cosine similarity to consider a face a match with a known person                                 |
| `NEW_PERSON_THRESHOLD`    |   `0.4` | If best similarity is below this value, the face is treated as unknown                                   |
| `UNKNOWN_MATCH_THRESHOLD` |  `0.55` | Similarity threshold used to determine whether an unknown face matches a previously tracked unknown face |
| `FRAMES_REQUIRED`         |     `8` | Number of frames an unknown face must appear in before it is added as a new person                       |
| `MAX_MISSES`              |    `10` | Maximum number of missed frames before an unknown tracked face is discarded                              |
| `MAX_EMBEDDINGS`          |    `20` | Maximum number of embeddings stored per known person                                                     |

---

# Database Schema

The project uses a SQLite database with one table:

## `persons`

| Column       | Type               | Description                             |
| ------------ | ------------------ | --------------------------------------- |
| `name`       | `TEXT PRIMARY KEY` | Unique identifier of the person         |
| `embeddings` | `TEXT`             | JSON-serialized list of face embeddings |

Each person may have multiple embeddings stored to improve matching reliability over time.

---

# Methods

## `load_db(self)`

Loads all persons and their stored embeddings from the SQLite database into memory.

### Returns

A list of dictionaries in the following format:

```python
[
    {
        "name": "person_name",
        "embeddings": [numpy_array_1, numpy_array_2, ...]
    }
]
```

---

## `save_person(self, name, embeddings)`

Saves or updates a person’s embeddings in the database.

### Parameters

* `name` (`str`): Person identifier
* `embeddings` (`list[np.ndarray]`): List of embeddings for that person

---

## `cosine_similarity(a, b)`

Static helper method used to compare two embeddings.

### Formula

```python
np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Parameters

* `a` (`np.ndarray`): First embedding
* `b` (`np.ndarray`): Second embedding

### Returns

* `float`: cosine similarity score between the two embeddings

Higher values mean the embeddings are more similar.

---

## `process_frame(self, frame)`

This is the main method of the system.
It processes a single video/image frame and returns detected face information.

### Parameters

* `frame`: image frame in OpenCV format (`numpy.ndarray`)

### Returns

A list of dictionaries like this:

```python
[
    {
        "bbox": [x1, y1, x2, y2],
        "label": "person_name (0.87)",
        "score": 0.87
    }
]
```

Where:

* `bbox` = bounding box coordinates of the detected face
* `label` = displayed identity label
* `score` = best similarity score found against known faces

---

# Recognition Logic

## 1. Match Against Known Database

For each detected face:

* The system extracts its embedding
* Compares it to **all saved embeddings of all known persons**
* Keeps the best match and best similarity score

If:

```python
best_score > MATCH_THRESHOLD
```

then the face is considered recognized.

### Result

* The face is labeled with the matched name
* The new embedding is appended to that person’s embedding history
* If the number of embeddings exceeds `MAX_EMBEDDINGS`, the oldest one is removed
* The updated embedding list is saved back to the database

---

## 2. Unknown Face Handling

If:

```python
best_score < NEW_PERSON_THRESHOLD
```

the face is treated as **unknown**.

The system then checks whether this unknown face matches one of the already tracked unknown faces using:

```python
UNKNOWN_MATCH_THRESHOLD
```

### If a tracked unknown face is matched:

* `seen_frames` is incremented
* `missed_frames` is reset
* The unknown face embedding is updated with a weighted average:

```python
matched_unknown["embedding"] = matched_unknown["embedding"] * 0.7 + emb * 0.3
```

This helps stabilize the embedding over time.

### If the unknown face has been seen enough times:

Once:

```python
seen_frames >= FRAMES_REQUIRED
```

a new person is created automatically:

* A new unique name is generated using `uuid`
* The embedding is saved to the database
* The person is added to the in-memory database

---

## 3. Ambiguous Matches

If the best similarity score falls between:

* `NEW_PERSON_THRESHOLD`
* `MATCH_THRESHOLD`

then the face is labeled as:

```python
"..."
```

This indicates the system is not confident enough to classify the face as known or completely unknown.

---

# Unknown Face Tracking

The class maintains an internal list:

```python
self.unknown_faces
```

Each tracked unknown face contains:

```python
{
    "embedding": emb,
    "seen_frames": 1,
    "missed_frames": 0,
    "updated": True
}
```

## Fields

* **embedding** — representative embedding for the unknown face
* **seen_frames** — how many frames the face has been matched in
* **missed_frames** — how many consecutive frames it was not seen
* **updated** — whether it was seen in the current frame

## Cleanup Logic

After processing a frame:

* Unknown faces not updated in the current frame have `missed_frames += 1`
* If an unknown face exceeds `MAX_MISSES`, it is removed from tracking

---

# Example Usage

## Basic frame processing

```python
import cv2
from face_recognition import FaceRecognition

cap = cv2.VideoCapture(0)
fr = FaceRecognition("faces.db")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = fr.process_frame(frame)

    for result in results:
        x1, y1, x2, y2 = result["bbox"]
        label = result["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
```

---

# Output Example

A sample result returned by `process_frame(frame)`:

```python
[
    {
        "bbox": [120, 80, 250, 230],
        "label": "a1b2c3d4e5f6 (0.78)",
        "score": 0.78
    },
    {
        "bbox": [300, 100, 420, 260],
        "label": "Unknown",
        "score": 0.21
    }
]
```

---

# Notes and Limitations

## 1. Auto-generated names

New persons are assigned a random UUID-based name:

```python
uuid.uuid4().hex[:12]
```

If you want human-readable names, you can replace this logic with your own naming strategy.

## 2. Potential issue with `label`

In the current implementation, when a new unknown face is first added to tracking, the `label` variable may not always be explicitly assigned before being appended to `results`. You may want to ensure a default label is set in every branch.

A safer pattern is to initialize:

```python
label = "Unknown"
```

before the recognition logic for each face.

## 3. Database growth

Each recognized person can accumulate up to `MAX_EMBEDDINGS` embeddings. This helps improve recognition quality but increases database size over time.

## 4. Performance

The current implementation compares every detected face embedding against every stored embedding. As the database grows, performance may degrade. For larger deployments, consider:

* approximate nearest neighbor search
* vector databases
* FAISS / Annoy / HNSW-based embedding indexing

## 5. GPU / runtime requirements

`InsightFace` may use GPU acceleration depending on your setup. If GPU is not available, inference will run on CPU and may be slower.

---

# Suggested Improvements

Here are a few practical enhancements you may want to add next:

* **Manual enrollment flow** for assigning real names to new people
* **Timestamp logging** for detections
* **Face image snapshots** for each registered person
* **Embedding normalization** before similarity comparison
* **Confidence smoothing** across frames
* **Track IDs** for more stable face tracking
* **Separate registration and recognition modes**
* **Threshold configuration from a config file**
* **REST API or GUI dashboard**

---

# Summary

This project provides a simple but effective face recognition pipeline with:

* persistent identity storage
* real-time matching
* automatic unknown-person registration
* incremental embedding updates
* frame-based unknown face tracking

It is a solid starting point for building a real-time face recognition system for attendance, access control, visitor tracking, or surveillance-style applications.
