import t2v_metrics

clip_flant5_score = t2v_metrics.VQAScore(model="clip-flant5-xxl")

# Single (image, text) pair demo
image = "/root/17753.png"
text = "a photo of a person talking on the phone angrily while another person sits happily"
score = clip_flant5_score(images=[image], texts=[text])
print("single score:", float(score[0, 0]))