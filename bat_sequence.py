"""
bat_sequence — ``####`` frame-sequence handling for the BAT loader.

A sequence is named by a single pattern string with a run of ``#`` standing in
for the zero-padded frame number (``plate_####.exr``). The run's LENGTH is the
padding, so ``####`` matches ``0101`` and not ``101`` — which is what keeps a
``shot_v002_####.exr`` from swallowing the version number.

Two paths reach the same set of files and must agree on it: find_sequence_files()
globs and filters, and scan_sequence_stats() fingerprints in one scandir pass.
Fingerprinting a different set of frames than the loader reads is worse than
fingerprinting slowly, so the regex is constructed identically in both.
"""

import glob
import logging
import os
import re
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

def _frame_run(pattern_path: str):
    """``(directory, before, padding, after)`` for the frame run of a pattern.

    The frame run is the LAST run of ``#`` in the file name — never one in the
    directory, which is just part of a folder's name. None when the file name
    has no ``#``. Both find_sequence_files() and scan_sequence_stats() split the
    pattern here, so they can't disagree about which files belong to it.
    """
    directory, filename = os.path.split(pattern_path.replace('\\', '/'))
    runs = list(re.finditer(r'#+', filename))
    if not runs:
        return None
    run = runs[-1]
    return (directory, filename[:run.start()], len(run.group(0)),
            filename[run.end():])


class SequenceHandler:
    """Locating and selecting the frames of a ``####`` sequence."""
    
    @staticmethod
    def detect_sequence_pattern(path: str) -> bool:
        """Whether `path` is a ``####`` pattern rather than a literal path.

        Only a run of ``#`` in the FILE NAME counts, and a path that exists as
        typed is never a pattern: ``/proj/Take #2/edit.mov`` or a file actually
        called ``clip#1.mp4`` is a real file, and treating its ``#`` as frame
        padding turned it into "no files match".
        """
        if not path:
            return False
        if not _frame_run(path):
            return False
        return not os.path.exists(path)
    
    @staticmethod
    def get_padding_from_template(template: str) -> int:
        """Extract padding length from #### pattern"""
        match = re.search(r'#+', template)
        if match:
            return len(match.group(0))
        return 4  # Default padding
    
    @staticmethod
    def replace_frame_number(filename: str, frame_number: int, padding_length: int = None) -> str:
        """Replace frame number in filename with proper padding"""
        if padding_length is None:
            padding_length = SequenceHandler.get_padding_from_template(filename)
        
        pattern = r'#+|\d+'
        padded_frame = str(frame_number).zfill(padding_length)
        
        def replacer(match):
            return padded_frame
        
        return re.sub(pattern, replacer, filename)
    
    @staticmethod
    def extract_frame_number_from_path(file_path: str) -> Optional[int]:
        """Extract frame number from a file path"""
        basename = os.path.basename(file_path)
        matches = re.findall(r'\d+', basename)
        
        if matches:
            for match in reversed(matches):
                if 3 <= len(match) <= 4:
                    return int(match)
            return int(matches[-1])
        return None
    
    @staticmethod
    def find_sequence_files(pattern_path: str) -> List[str]:
        """Find all files matching the sequence pattern"""
        # Normalize slashes for consistency
        pattern_path = pattern_path.replace('\\', '/')

        parts = _frame_run(pattern_path)
        if parts is None:
            return []
        directory, before, padding_len, after = parts
        prefix = directory.rstrip("/") + "/" if directory else ""

        # Everything but the frame run is escaped, for glob and regex alike: a
        # folder called `shots[v2]` is a character class to an unescaped glob,
        # which then matched nothing while the one-pass scan (a literal
        # scandir) counted every frame.
        glob_pattern = glob.escape(prefix + before) + '*' + glob.escape(after)
        matching_files = [f.replace('\\', '/') for f in glob.glob(glob_pattern)]

        # Create regex for exact match
        regex_pattern = re.compile(
            "^" + re.escape(prefix + before) + rf'\d{{{padding_len}}}'
            + re.escape(after) + "$", re.IGNORECASE)

        valid_files = [f for f in matching_files if regex_pattern.match(f)]
        logger.debug("[Bat_Loader] found %d sequence files for %s",
                     len(valid_files), pattern_path)

        return sorted(valid_files)

    @staticmethod
    def scan_sequence_stats(pattern_path: str) -> Optional[Tuple[int, int, int]]:
        """One-pass (frame count, max mtime_ns, total size) for a #### pattern.

        This is the fingerprint IS_CHANGED needs, and IS_CHANGED runs for every
        node in a queued prompt — ComfyUI computes cache keys over all of
        prompt.keys(), not just the nodes it's going to execute — so a
        loader that isn't wired to anything still pays it on every submit.
        Globbing and then os.stat()ing each frame walks the directory twice and
        issues a separate attribute lookup per file; across NFS a few thousand
        EXRs make that a visible stall before the run even starts. Scanning once
        keeps the attribute reads inside the directory listing that already
        fetched them (readdirplus), so the sequence is fingerprinted for roughly
        the cost of listing it.

        Returns None when the fast path doesn't apply or the scan breaks, which
        means "fall back to the glob path" rather than "no files".
        """
        parts = _frame_run(pattern_path)
        if parts is None:
            return None
        directory, before, padding_len, after = parts
        if not directory:
            return None

        # Same construction as find_sequence_files(), so the two agree on which
        # files belong to the sequence — a fingerprint over a different set of
        # frames than the loader reads is worse than a slow one.
        pattern_for_regex = (re.escape(before) + rf'\d{{{padding_len}}}'
                             + re.escape(after))
        # Case-sensitive on purpose: find_sequence_files() reaches the same set
        # through a case-sensitive glob (its IGNORECASE regex only ever narrows
        # what the glob already returned), so matching case here keeps the
        # fingerprint over exactly the frames the loader will read.
        regex_pattern = re.compile(f"^{pattern_for_regex}$")

        count = 0
        max_mtime_ns = 0
        total_size = 0
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not regex_pattern.match(entry.name):
                        continue
                    st = entry.stat()
                    count += 1
                    if st.st_mtime_ns > max_mtime_ns:
                        max_mtime_ns = st.st_mtime_ns
                    total_size += st.st_size
        except OSError:
            return None

        return (count, max_mtime_ns, total_size)

    @staticmethod
    def extract_frame_numbers(file_paths: List[str]) -> List[Tuple[int, str]]:
        """Extract frame numbers from file paths"""
        frame_info = []
        for file_path in file_paths:
            frame_num = SequenceHandler.extract_frame_number_from_path(file_path)
            if frame_num is not None:
                frame_info.append((frame_num, file_path))
        frame_info.sort()
        return frame_info
    
    @staticmethod
    def select_sequence_frames(available_frames: List[Tuple[int, str]], start_frame: int, end_frame: int, frame_step: int) -> List[str]:
        """Select specific frames from available sequence"""
        selected_frames = []
        current_frame = start_frame
        while current_frame <= end_frame:
            found = False
            for frame_num, file_path in available_frames:
                if frame_num == current_frame:
                    selected_frames.append(file_path)
                    found = True
                    break
            if not found:
                selected_frames.append(None)
            current_frame += frame_step
        return selected_frames
    
    @staticmethod
    def validate_sequence_parameters(start_frame: int, end_frame: int, frame_step: int) -> Tuple[int, int, int]:
        """Validate and sanitize sequence parameters"""
        start_frame = start_frame or 0
        end_frame = end_frame or start_frame
        frame_step = frame_step or 1
        return max(0, start_frame), max(start_frame, end_frame), max(1, frame_step)