CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    frame_no INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    seam_type TEXT NOT NULL CHECK(seam_type IN ('U', 'B')),
    similarity REAL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
    image_path TEXT,
    col2_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labels_status ON labels(status);
CREATE INDEX IF NOT EXISTS idx_labels_name ON labels(name);
