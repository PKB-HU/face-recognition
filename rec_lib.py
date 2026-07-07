import cv2
import numpy as np
import sqlite3
import json
from insightface.app import FaceAnalysis
import uuid


class FaceRecognition:
    def __init__(self, db_file="faces.db"):
        self.db_file = db_file
        self.MATCH_THRESHOLD = 0.6
        self.NEW_PERSON_THRESHOLD = 0.4
        self.UNKNOWN_MATCH_THRESHOLD = 0.55
        self.FRAMES_REQUIRED = 8
        self.MAX_MISSES = 10
        self.MAX_EMBEDDINGS = 20

        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.c = self.conn.cursor()
        self.c.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            name TEXT PRIMARY KEY,
            embeddings TEXT
        )
        """)
        self.conn.commit()

        self.db = self.load_db()
        self.unknown_faces = []
        self.person_counter = len(self.db)
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0)

    def load_db(self):
        self.c.execute("SELECT name, embeddings FROM persons")
        db = []
        for name, emb_json in self.c.fetchall():
            db.append({
                "name": name,
                "embeddings": [np.array(e) for e in json.loads(emb_json)]
            })
        return db

    def save_person(self, name, embeddings):
        embeddings_json = json.dumps([e.tolist() for e in embeddings])
        self.c.execute("""
        INSERT OR REPLACE INTO persons (name, embeddings)
        VALUES (?, ?)
        """, (name, embeddings_json))
        self.conn.commit()

    #azonosság
    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def process_frame(self, frame):
        faces = self.app.get(frame)
        results = []

        for u in self.unknown_faces:
            u["updated"] = False

        for face in faces:
            emb = face.embedding
            best_match = None
            best_score = -1

            #adatbázissal matchlees
            for person in self.db:
                for saved_emb in person["embeddings"]:
                    score = self.cosine_similarity(emb, np.array(saved_emb))
                    if score > best_score:
                        best_score = score
                        best_match = person["name"]

            x1, y1, x2, y2 = map(int, face.bbox)

            if best_score > self.MATCH_THRESHOLD:
                label = f"{best_match} ({best_score:.2f})"
                for person in self.db:
                    if person["name"] == best_match:
                        person["embeddings"].append(emb)
                        if len(person["embeddings"]) > self.MAX_EMBEDDINGS:
                            person["embeddings"].pop(0)
                        self.save_person(person["name"], person["embeddings"])

            elif best_score < self.NEW_PERSON_THRESHOLD:
                label = "Unknown"
                matched_unknown = None
                for u in self.unknown_faces:
                    sim = self.cosine_similarity(emb, u["embedding"])
                    if sim > self.UNKNOWN_MATCH_THRESHOLD:
                        matched_unknown = u
                        break

                if matched_unknown:
                    matched_unknown["seen_frames"] += 1
                    matched_unknown["missed_frames"] = 0
                    matched_unknown["updated"] = True
                    matched_unknown["embedding"] = (
                            matched_unknown["embedding"] * 0.7 + emb * 0.3
                    )

                    if matched_unknown["seen_frames"] >= self.FRAMES_REQUIRED:
                        self.person_counter += 1
                        new_name = uuid.uuid4().hex[:12]

                        self.db.append({
                            "name": new_name,
                            "embeddings": [matched_unknown["embedding"]]
                        })
                        self.save_person(new_name, [matched_unknown["embedding"]])
                        matched_unknown["label"] = new_name
                        self.unknown_faces.remove(matched_unknown)
                else:
                    self.unknown_faces.append({
                        "embedding": emb,
                        "seen_frames": 1,
                        "missed_frames": 0,
                        "updated": True
                    })
            else:
                label = "..."


            results.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "score": float(best_score)
            })

        for u in self.unknown_faces:
            if not u["updated"]:
                u["missed_frames"] += 1
            else:
                u["missed_frames"] = 0
        self.unknown_faces = [u for u in self.unknown_faces if u["missed_frames"] < self.MAX_MISSES]

        return results