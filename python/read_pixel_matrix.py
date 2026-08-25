from pathlib import Path

from PIL import Image


def read_pixel_matrix(image_path: Path) -> list[list[tuple[int, ...]]]:
    """Return the image pixels as a row-major matrix of RGBA tuples."""
    with Image.open(image_path) as image:
        rgba_image = image.convert("RGBA")
        width, height = rgba_image.size
        pixels = list(rgba_image.get_flattened_data())

    return [pixels[row * width : (row + 1) * width] for row in range(height)]


if __name__ == "__main__":
    image_path = Path(__file__).parent.parent / "resources" / "ABCMO_294_detail.png"
    pixel_matrix = read_pixel_matrix(image_path)
    print(f"Loaded {len(pixel_matrix)} rows x {len(pixel_matrix[0])} columns.")