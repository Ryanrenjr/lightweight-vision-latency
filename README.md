Lightweight Image Classification: Accuracy vs Speed 

I. What is this project about?

In many real situations (robotics, mobile apps, AR, etc.), a vision model must do two things:

Be accurate.

Run fast in real time.

In this project I try to understand that trade-off.

I did two main things:

I fine-tuned a small image classification model (ResNet-18) on the CIFAR-10 dataset, so it can recognize 10 common classes (cat, dog, car, airplane, etc.).

I measured how fast this model runs on my own laptop, and I looked at the images it got wrong to understand why it failed.

This is close to real deployment thinking. It is not only “get a high accuracy number”, but “will this model actually work in the real world?”


II. Environment / setup

·My machine: Apple MacBook (macOS) ·Python version: 3.12.0 ·Main libraries: ·torch 2.2.2 ·torchvision 0.17.2 ·numpy 1.26.4 ·opencv-python 4.7.0 ·Dataset: CIFAR-10 ·10 classes of common objects/animals ·each image is tiny (32×32 RGB) ·50,000 train images, 10,000 test images

I ran everything in a Python virtual environment: python -m venv .venv source .venv/bin/activate

III. Training phase (train.py)

What this script does:

Load a pretrained ResNet-18 model. (ResNet-18 is a classic small CNN.)
Replace the last layer so it outputs 10 classes (for CIFAR-10).
Fine-tune on CIFAR-10 for a few epochs.
After each epoch, print training loss and training accuracy.
Test on the test set and print the final test accuracy.
Save the trained weights to a file called resnet18_cifar10.pt.
Concept idea (not full code):

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT) model.fc = nn.Linear(model.fc.in_features, 10)

train the model...
torch.save(model.state_dict(), "resnet18_cifar10.pt")

Real output from my machine: Epoch 1: train loss 1.0257, acc 64.69% Epoch 2: train loss 0.7380, acc 74.99% Epoch 3: train loss 0.6538, acc 77.78% Epoch 4: train loss 0.6044, acc 79.26% Epoch 5: train loss 0.5473, acc 81.22% Test accuracy: 80.88% Saved: resnet18_cifar10.pt

What this means:

After only a short fine-tune, the model reaches about 80.88% test accuracy.
Now I have a working small vision model and a weight file I can load again later.


IV. Inference speed / latency test (bench.py)

Why I do this: In real products (robot, camera feed, AR app), accuracy is not enough. The model must be fast. We need to know: how many milliseconds per image? How many frames per second?

What bench.py does:

Load the saved weights resnet18_cifar10.pt.

Take 200 test images.

Run inference one image at a time (batch size = 1). This simulates real-time video, frame by frame.

Measure latency per frame in milliseconds.

Report:

average time (avg)

median time (p50)

95th percentile slow time (p95)

FPS (frames per second = how many images per second)

Do this on:

CPU

Apple Metal backend (mps) on my Mac, which is like a GPU-style backend.

Real output from my run:

[cpu] n=200 avg=4.3ms p50=4.3ms p95=5.4ms FPS=230.6 [mps] n=200 avg=11.9ms p50=11.8ms p95=14.2ms FPS=83.8

How to read this:

-On my machine, the CPU can run one image in ~4.3 ms (around 230 FPS). -On Apple mps, it was ~11.9 ms per image (~83.8 FPS).

This looks “backwards” because usually we think “GPU is faster.” But here:

-We only infer 1 image at a time (batch=1). -The model (ResNet-18) is small. -Sending data to GPU / mps has overhead. -The Apple mps backend also has some launch cost.

So in this real-time, single-frame scenario, CPU is actually faster. This shows why you cannot assume “GPU is always better.” You must test for your real use case.



V. Misclassified samples / error analysis (miscls.py)

It is not enough to say “accuracy is 80.88%”. I also need to see where it fails.

What miscls.py does:

Run the trained model on the CIFAR-10 test set.
Find images where the prediction is wrong.
Convert those images back to normal RGB (undo normalization).
Add text on the image saying: -pred: (what the model predicted) -gt: (the true label / ground truth)
Save a few of these wrong examples into the miscls/ folder. Real output from my run: saved: miscls/sample_0035.jpg -> pred: automobile gt: bird saved: miscls/sample_0042.jpg -> pred: cat gt: dog saved: miscls/sample_0053.jpg -> pred: dog gt: cat Done. Saved 3 samples into ./miscls/ How I interpret these: -bird → automobile: CIFAR-10 images are tiny (32×32). The “bird” is far away and blurry. The background has shiny edges / bright shapes, so the model thinks it is a car body / reflection. -cat ↔ dog: At 32×32, face and fur details are very low resolution. The model mostly sees rough texture and outline, and small pets can look similar, so it confuses cat and dog.
Why this is useful: -I am not just giving one big number. I can explain why the model is wrong. -I can also propose improvements, for example: -Use higher input resolution (for example upsample to 48×48 or 64×64 before feeding the model). -Use stronger data augmentation (blur, color jitter, cutout) so the model learns to be robust to bad quality images. -Tune batch size / hardware choice to keep a good balance between accuracy and latency.

This is important for real-world vision and robotics, because real camera images are often blurry, dark, or partly blocked. They are not perfect “textbook” images.



VI. Summary table (accuracy + speed) Setup / Scenario Top-1 Acc. Avg (ms) p50 (ms) p95 (ms) FPS CPU inference (ResNet-18, batch=1) 80.88% 4.3 4.3 5.4 230.6 MPS (Apple Metal backend, batch=1) 80.88% 11.9 11.8 14.2 83.8

Notes: -The model and weights are the same. Only the runtime backend changes. -CPU wins here because the model is small and we use batch size = 1. -For bigger models or bigger batches, a GPU / MPS backend can be faster. -This thinking (accuracy + latency + stability) is important for robotics and any edge/real-time system.

VII.What I learned (and why it matters for MSc applications)

End-to-end ability I did everything myself: environment setup, fine-tuning the model, saving weights, loading the model again, benchmarking speed, and doing error analysis. This shows I can run a small computer vision pipeline from start to finish, not just copy a notebook.

Engineering mindset (not only theory) I did real latency tests (avg / p50 / p95) and FPS on my own hardware. I compared CPU vs MPS and explained why CPU can be faster in this case. This is deployment thinking: “Will this model actually run fast enough in the real world?”

Reliability and failure cases I did not stop at “80.88% accuracy”. I exported real wrong predictions (cat↔dog, bird→car), and I explained why the model was confused (low resolution, look-alike classes, background noise). I also suggested how to improve (higher resolution, stronger data augmentation, etc.). This shows I care about robustness in real conditions, not just about getting a nice number.

In other words, this project is not just “I can code in PyTorch.” It shows I understand how to push a vision model toward real deployment and how to talk about its limits.



VIII. Project files

train.py Fine-tunes ResNet-18 on CIFAR-10 and saves the weights.

bench.py Loads the trained model and measures speed / latency / FPS on CPU and MPS.

miscls.py Finds misclassified test images, writes predicted label and true label on top of them, and saves them in a folder.

resnet18_cifar10.pt The trained model weights.

miscls/ Folder with example “wrong predictions” images (for error analysis).

README.md, README_simple_english.md Documentation of what I did and what I learned.

All numbers and screenshots in this document were produced by me (Jiexin Ren) on my own Mac in October 2025.
