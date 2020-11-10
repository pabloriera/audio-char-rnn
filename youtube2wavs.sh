set -xe

sr=8000
codec='pcm_u8'

pip -q install youtube-dl
mkdir -p wavs/8bit

c=0
for link in https://www.youtube.com/watch?v=QobNvudIGGY  https://www.youtube.com/watch?v=nSmCGK2ZzFs; do
	c=$((c+1))
    fname=wavs/$c.wav
    fname8=wavs/8bit/$c.wav
    youtube-dl --extract-audio --audio-format wav --output $fname $link
    ffmpeg -y -i $fname -ar $sr -ac 1 -acodec $codec $fname8
done