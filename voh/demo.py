import time

from .utils import *
from .voh import *


def demo(model, ref=None, target=None, threshold=0.7, dur=4, osa=True):
    rec = record_voice_osa if osa else record_voice
    while True:
        if ref is None:
            sound_rec_voice1()
            voice1 = rec(tmpfile(suffix=".wav"), dur=dur)
            time.sleep(0.3)
        else:
            sound_voice1()
            voice1 = play(ref)

        if target is None:
            sound_rec_voice2()
            voice2 = rec(tmpfile(suffix=".wav"), dur=dur)
            time.sleep(0.3)
        else:
            sound_voice2()
            voice2 = play(target)

        if model.verify(voice1, voice2, threshold=threshold):
            sound_same()
            break
        else:
            sound_different()
            break


# ----------------------
# Voice recording
# ----------------------
def sound_voice1():
    play("assets/voice1.wav")


def sound_voice2():
    play("assets/voice2.wav")


def sound_rec_voice1():
    play("assets/rec-voice1.wav")


def sound_rec_voice2():
    play("assets/rec-voice2.wav")


def sound_tik():
    play("assets/tik.wav")


def sound_tok():
    play("assets/tok.wav")


def sound_same():
    play("assets/same.wav")


def sound_different():
    play("assets/different.wav")


def record_voice(out, dur=5, sr=44100):
    sound_tik()
    audio = sd.rec(int(dur * sr), samplerate=sr, channels=1)
    sd.wait()
    sound_tok()
    return savewav(out, (audio, sr))


def record_voice_osa(out=None, dur=5):
    d = rf"{HOME()}/Library/Group\ Containers/group.com.apple.VoiceMemos.shared/Recordings"
    script = tmpfile(suffix=".scpt")
    writer(script).write(
        f"""
    tell application "/System/Applications/VoiceMemos.app"
        activate
    end tell

    delay 0.3

    tell application "System Events"
        keystroke "n" using {{command down}}
        delay {dur}
    end tell

    tell application "System Events"
        tell process "Voice Memos"
            click menu item "Done Editing" of menu "File" of menu bar 1
            click menu item "Hide Voice Memos" of menu "Voice Memos" of menu bar 1
        end tell
    end tell"""
    )

    out = out or tmpfile(suffix=".wav")
    shell(f"osascript {script}")
    m4a = fst(shell(f'ls -t "{d}/*.m4a" | head -n 1'))
    shell(rf"{_ffmpeg} -i \"{m4a}\" {out}")
    return out
