from diffusers import StableDiffusionPipeline


def main() -> None:
    ACCESS_TOKEN = "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"
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", use_auth_token=ACCESS_TOKEN
    )
    prompt = "cute cat paly with ball"
    image = pipe(prompt)["sample"][0]
    image.save("cat.png")


if __name__ == "__main__":
    main()
