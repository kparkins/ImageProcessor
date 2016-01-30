#include "image.h"
#include "bmp.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <functional>


/**
 * Image
 **/
Image::Image (int width_, int height_)
{
    assert(width_ > 0);
    assert(height_ > 0);

    width           = width_;
    height          = height_;
    num_pixels      = width * height;
    pixels          = new Pixel[num_pixels];
    sampling_method = IMAGE_SAMPLING_POINT;

    assert(pixels != NULL);
}


Image::Image (const Image& src)
{
    width           = src.width;
    height          = src.height;
    num_pixels      = width * height;
    pixels          = new Pixel[num_pixels];
    sampling_method = IMAGE_SAMPLING_POINT;

    assert(pixels != NULL);
    memcpy(pixels, src.pixels, src.width * src.height * sizeof(Pixel));
}

Image::Image() {

}


Image::~Image ()
{
    delete [] pixels;
    pixels = NULL;
}

/*
void Image::AddNoise (double factor)
{

}
*/

void Image::Brighten (double factor)
{
  /* Your Work Here  (section 3.2.1 of assignment)*/
  if(factor < 0.f) {
    std::cerr << "Error. Brighten factor less than 0." << std::endl;
    return;
  }

  float r = 0.f;
  float g = 0.f;
  float b = 0.f;
  for(int i = 0; i < num_pixels; ++i) {
    r = (float) pixels[i].r * factor;
    g = (float) pixels[i].g * factor;
    b = (float) pixels[i].b * factor;
    pixels[i].SetClamp(r, g, b);
  }

}


void Image::ChangeContrast (double factor)
{
  float luminance = 0;
  for(int i = 0; i < num_pixels; ++i) {
    luminance += pixels[i].Luminance();
  }

  bool invert = false;
  if(factor < 0) {
    invert = true;
    factor *= -1.0;
  }
  luminance /= num_pixels;

  for(int i = 0; i < num_pixels; ++i) {
    pixels[i].r = ComponentLerp(luminance, pixels[i].r, factor);
    pixels[i].g = ComponentLerp(luminance, pixels[i].g, factor);
    pixels[i].b = ComponentLerp(luminance, pixels[i].b, factor);

    if(invert) {
      pixels[i].Set(255 - pixels[i].r, 255 - pixels[i].g, 255 - pixels[i].b);
    }
  }
}


void Image::ChangeSaturation(double factor)
{
  bool invert = false;
  if(factor < 0) {
    invert = true;
    factor *= -1.0;
  }

  for(int i = 0; i < num_pixels; ++i) {
    float greyColor = (pixels[i].r + pixels[i].g + pixels[i].b) / 3.f; 

    pixels[i].r = ComponentLerp(greyColor, pixels[i].r, factor);
    pixels[i].g = ComponentLerp(greyColor, pixels[i].g, factor);
    pixels[i].b = ComponentLerp(greyColor, pixels[i].b, factor);

    if(invert) {
      pixels[i].Set(255 - pixels[i].r, 255 - pixels[i].g, 255 - pixels[i].b);
    }
  }
}

void Image::ChangeGamma(double factor)
{
  if(factor < 0.0) {
    std::cerr << "Error. Negative gamma correction value. Value must be "
              << "greater than or equal to 0." 
              << std::endl;
    return;
  }

  for(int i = 0; i < num_pixels; ++i) {
    pixels[i].SetClamp(pow((float)pixels[i].r / 256.f, 1.f / factor) * 256.f, 
                       pow((float)pixels[i].g / 256.f, 1.f / factor) * 256.f, 
                       pow((float)pixels[i].b / 256.f, 1.f / factor) * 256.f);
  }
}

Image* Image::Crop(int x, int y, int w, int h)
{
  if(x < 0 || y < 0 || w <= 0 || h <= 0) {
    std::cerr << "Error. X & Y must be >= 0. W & H must be > 0." << std::endl;
    return NULL;
  }

  if(x >= width || y >= height) {
    std::cerr << "Error. X or Y value out of image pixel ranged." << std::endl;
    return NULL;
  }

  if(x + w > width) {
    std::cerr << "Crop width greater than image width. "
              << "Capping crop width at image width." << std::endl;
    w = width - x;
  }

  if(y + h > height) {
    std::cerr << "Crop height greater than image height. "
              << "Capping crop height at image height." << std::endl;
    h = height - y;
  }

  int dy = 0;
  Image* image = new Image(w, h);
  assert(image != NULL);
  Pixel* targetPixels = image->pixels;
  for(int sy = y; sy < height && sy - y < h; ++sy, ++dy) {
    int dx = 0;
    for(int sx = x; sx < width && sx - x < w; ++sx, ++dx) {
      targetPixels[dy * w + dx] = pixels[sy * width + sx];
    }
  }
  return image;
}

/*
void Image::ExtractChannel(int channel)
{
  // For extracting a channel (R,G,B) of image.  
  // Not required for the assignment
}
*/

static Component QuantizeChannel(Component original, int numChannels, float noise = 0) {
  float normalized = ((float) original) / 256.f;
  float quantum = floor(normalized * numChannels + noise);
  if(quantum < 0) {
    quantum = 0;
  } else if(quantum > numChannels - 1) {
    quantum = numChannels - 1;
  }
  return (Component) (floor(255.f * quantum / (numChannels - 1)));
}

void Image::Quantize (int nbits)
{
  if(nbits < 1 || nbits > 8) {
    std::cerr << "Error. Quantize only accepts arguments in the range [1, 8]." << std::endl;
    return;
  }

  int numChannels = pow(2, nbits);
  
  for(int i = 0; i < num_pixels; ++i) {
    Pixel& p = pixels[i];
    p.r = QuantizeChannel(p.r, numChannels);
    p.g = QuantizeChannel(p.g, numChannels);
    p.b = QuantizeChannel(p.b, numChannels);
  }
}

void Image::RandomDither (int nbits)
{
  srand(time(NULL));
  if(nbits < 1 || nbits > 8) {
    std::cerr << "Error. Random Dither only accepts arguments in the range [1, 8]." << std::endl;
    return;
  }

  int numChannels = pow(2, nbits);
  
  for(int i = 0; i < num_pixels; ++i) {
    float noise = ((float) rand() / (float) RAND_MAX) - .5;
    Pixel& p = pixels[i];
    p.r = QuantizeChannel(p.r, numChannels, noise);
    p.g = QuantizeChannel(p.g, numChannels, noise);
    p.b = QuantizeChannel(p.b, numChannels, noise);
  }
}


/* Matrix for Bayer's 4x4 pattern dither. */
/* uncomment its definition if you need it */

/*
static int Bayer4[4][4] =
{
    {15, 7, 13, 5},
    {3, 11, 1, 9},
    {12, 4, 14, 6},
    {0, 8, 2, 10}
};


void Image::OrderedDither(int nbits)
{
  // For ordered dithering
  // Not required for the assignment
}

*/
static float QuantizeFloyd(Component original, int numChannels) {
  float normalized = ((float) original) / 256.f;
  float quantum = floor(normalized * numChannels);
  return (float) (floor(255.f * quantum / (numChannels - 1)));
}

static float clamp(float min, float max, float value) {
  if(value < min) {
    return min;
  }
  if(value > max) {
    return max;
  }
  return value;
}

/* Error-diffusion parameters for Floyd-Steinberg*/
const double
    ALPHA = 7.0 / 16.0,
    BETA  = 3.0 / 16.0,
    GAMMA = 5.0 / 16.0,
    DELTA = 1.0 / 16.0;

int Image::ValidX(int x) {
  return mod<int>(x, width);
}

int Image::ValidY(int y) {
  return mod<int>(y, height);
}

Pixel& Image::GetValidPixel(int x, int y) {
  return pixels[mod<int>(y, height) * width + mod<int>(x, width)];
}

void Image::UpdatePixelError(int x, int y, float r, float g, float b, double weight) {
  Pixel & p = GetValidPixel(x,y);
  p.r = clamp(0.f, 255.f, ((float) p.r) + r * weight);
  p.g = clamp(0.f, 255.f, ((float) p.g) + g * weight);
  p.b = clamp(0.f, 255.f, ((float) p.b) + b * weight);
}

void Image::FloydSteinbergDither(int nbits)
{
  if(nbits < 1 || nbits > 8) {
    std::cerr << "Error. Floyd Steinberg Dither only accepts arguments in the range [1, 8]." << std::endl;
    return;
  }
  float r = 0;
  float g = 0;
  float b = 0;
  int numChannels = pow(2, nbits);
  for(int j = 0; j < height; ++j) {
    for(int i = 0; i < width; ++i) {
      Pixel & p = pixels[j * width + i]; 

      r = ((float) p.r) - QuantizeFloyd(p.r, numChannels);
      g = ((float) p.g) - QuantizeFloyd(p.g, numChannels);
      b = ((float) p.b) - QuantizeFloyd(p.b, numChannels);

      p.r = clamp(0, 255, QuantizeFloyd(p.r, numChannels));
      p.g = clamp(0, 255, QuantizeFloyd(p.g, numChannels));
      p.b = clamp(0, 255, QuantizeFloyd(p.b, numChannels));

      UpdatePixelError(i + 1, j    , r, g, b, ALPHA);  
      UpdatePixelError(i - 1, j + 1, r, g, b, BETA);
      UpdatePixelError(i    , j + 1, r, g, b, GAMMA);
      UpdatePixelError(i + 1, j + 1, r, g, b, DELTA);
    }
  }
}

void ImageComposite(Image *bottom, Image *top, Image *result)
{
  // Extra Credit (Section 3.7).
  // This hook just takes the top image and bottom image, producing a result
  // You might want to define a series of compositing modes as OpenGL does
  // You will have to use the alpha channel here to create Mattes
  // One idea is to composite your face into a famous picture
}

void Image::Convolve(int *filter, int n, int normalization) {
  int d = (n - 1) / 2;
  float r = 0;
  float g = 0;
  float b = 0;
  float weight = 0;

  if(d <= 0) {
    std::cerr << "Kernel width invalid. Copying original image." << std::endl;
    return;
  }

  Pixel* resultPixels = new Pixel[width * height];
  for(int j = 0; j < height; ++j) {
    for(int i = 0; i < width; ++i) {
      r = 0;
      g = 0;
      b = 0;
      // Apply filter to local area centered at ij
      for(int dy = -d; dy <= d; ++dy) {
        for(int dx = -d; dx <= d; ++dx) {
          // GetValidPixel automatically handles edge case
          // using "toroidal" picture
          Pixel & p = GetValidPixel(i + dx, j + dy);
          weight = filter[(dy + d) * n + (dx + d)]; 
          r += weight * (float) p.r;
          g += weight * (float) p.g;
          b += weight * (float) p.b;
        }
      }  
      r /= (float) normalization;
      g /= (float) normalization;
      b /= (float) normalization;
      resultPixels[j * width + i].SetClamp(r,g,b);
    }
  }

  delete pixels;
  pixels = resultPixels;
}

float Gaussian(int u, int v, float n) {
  float sigma = floor(n / 2.f) / 2.f;
  return (1.f / (2.f * M_PI * sigma * sigma)) * exp(-(u * u + v * v) / (2.f * sigma * sigma));
}

void Image::Blur(int n)
{
  if(n < 1 || n % 2 == 0) {
    std::cerr << "Error. Filter width to Image::Blur must be a positive, odd integer." << std::endl;
    return;
  }

  int halfWidth = (n - 1) / 2;
  float smallest = FLT_MAX;
  float tempKernel[n * n];
  // make a temporary kernel (of floats)
  for(int j = 0; j < n; ++j) {
    for(int i = 0; i < n; ++i) {
      tempKernel[j * n + i] = Gaussian(i - halfWidth, j - halfWidth, n);
      // find smallest element while building float kernel
      if(tempKernel[j * n + i] < smallest) {
        smallest = tempKernel[j * n + i];
      }
    }
  } 

  int kernel[n * n];
  int normalization = 0;
  // convert the float kernel into an int kernel
  for(int j = 0; j < n; ++j) {
    for(int i = 0; i < n; ++i) {
      kernel[j * n + i] = floor(tempKernel[j * n + i] / smallest); 
      normalization += kernel[j * n + i];
    }
  }
  
  this->Convolve(kernel, n, normalization);
}

void Image::Sharpen() 
{
  static int kernel[9] = {
    -1, -2, -1, 
    -2, 19, -2, 
    -1, -2, -1
  };
  this->Convolve(kernel, 3, 7);
}

void Image::ConvolveEdgeDetect(PixelFloat* result, int* filter, int n) {
  int d = (n - 1) / 2;
  float r = 0;
  float g = 0;
  float b = 0;
  float weight = 0;

  if(d <= 0) {
    std::cerr << "Kernel width invalid. Copying original image." << std::endl;
    return;
  }

  for(int j = 0; j < height; ++j) {
    for(int i = 0; i < width; ++i) {
      r = 0;
      g = 0;
      b = 0;
      // Apply filter to local area centered at ij
      for(int dy = -d; dy <= d; ++dy) {
        for(int dx = -d; dx <= d; ++dx) {
          // GetValidPixel automatically handles edge case
          // using "toroidal" picture
          Pixel& p = GetValidPixel(i + dx, j + dy);
          weight = filter[(dy + d) * n + (dx + d)]; 
          r += weight * (float) p.r;
          g += weight * (float) p.g;
          b += weight * (float) p.b;
        }
      }  
      result[j * width + i].Set(r,g,b);
    }
  }
}

void Image::EdgeDetect(int threshold)
{
  static int fx[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
  };

  static int fy[9] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1
  };

  PixelFloat* gx = new PixelFloat[width * height];
  PixelFloat* gy = new PixelFloat[width * height];
  Pixel* g = new Pixel[width * height];

  this->ConvolveEdgeDetect(gx, fx, 3);
  this->ConvolveEdgeDetect(gy, fy, 3);

  float lx, ly;
  for(int i = 0; i < num_pixels; ++i) {
    lx = gx[i].Luminance();
    ly = gy[i].Luminance();
    if(lx * lx + ly * ly >= threshold * threshold) {
      g[i].SetClamp(255, 255, 255);
    } else {
      g[i].SetClamp(0, 0, 0);
    }
  }

  delete pixels;
  pixels = g;
}

float HatWeight(float x) {
  if(fabs(x) > 1.0) {
    return 0.0;
  }
  return 1.0 - fabs(x);
} 

float MitchellWeight(float x) {
  x = fabs(x);
  if(x >= 1.f && x < 2.f) {
    return (1.f / 6.f) * (((-7.f/3.f) * x * x * x) + (12 * x * x) - (20 * x) + (32.f / 3.f)); 
  } else if(x >= 0.f && x < 1.f){
    return (1.f / 6.f) * ((7.f * x * x * x) - (12 * x * x) + (16.f / 3.f));
  }
  return 0.f;
}

void MagnifyX(Image* dst, Image* src, int smethod) {
  float x0 = 0.f;
  float weight = 0.f;
  float width = (smethod == IMAGE_SAMPLING_HAT) ? 2.f : 4.f;
  float normalization = 0.f;
  Pixel* result = dst->pixels;
  float s = (float) dst->width / (float) src->width;
  PixelFloat pf;

  for(int j = 0; j < dst->height; ++j) {
    for(int i = 0; i < dst->width; ++i) {
      if(smethod == IMAGE_SAMPLING_POINT) {
        result[j * dst->width + i] = src->GetValidPixel((float) i / ((float) dst->width - 1.f) * src->width, j);
        continue;
      }
      memset((void*) &pf.r, 0, sizeof(pf));
      normalization = 0.f;
      x0 = (float) i / s;
      for(int x = x0 - width; x <= x0 + width; ++x) {
        if(smethod == IMAGE_SAMPLING_HAT) {
          weight = HatWeight(x - i / s);
        } else if(smethod == IMAGE_SAMPLING_MITCHELL) {
          weight = MitchellWeight(x - i / s);
        }
        Pixel & p = src->GetValidPixel(x, j);
        pf.r += weight * (float) p.r;
        pf.g += weight * (float) p.g;
        pf.b += weight * (float) p.b;
        normalization += weight;
      }
      pf.r /= normalization;
      pf.g /= normalization;
      pf.b /= normalization;
      result[j * dst->width + i].SetClamp(pf.r, pf.g, pf.b);
    }
  }
}

void MinifyX(Image* dst, Image* src, int smethod) {
  float x0 = 0.f;
  float weight = 0.f;
  float normalization = 0.f;
  Pixel* result = dst->pixels;
  float s = (float) dst->width / (float) src->width;
  float width = (smethod == IMAGE_SAMPLING_HAT) ? 2.f / s: 4.f / s;
  PixelFloat pf;

  for(int j = 0; j < dst->height; ++j) {
    for(int i = 0; i < dst->width; ++i) {
      if(smethod == IMAGE_SAMPLING_POINT) {
        result[j * dst->width + i] = src->GetValidPixel((float) i / ((float) dst->width - 1.f) * src->width, j);
        continue;
      }
      memset((void*) &pf.r, 0, sizeof(pf));
      normalization = 0.f;
      x0 = (float) i / s;
      for(int x = x0 - width; x <= x0 + width; ++x) {
        if(smethod == IMAGE_SAMPLING_HAT) {
          weight = HatWeight(x * s - i);
        } else if(smethod == IMAGE_SAMPLING_MITCHELL) {
          weight = MitchellWeight(x * s - i);
        }
        Pixel & p = src->GetValidPixel(x, j);
        pf.r += weight * (float) p.r;
        pf.g += weight * (float) p.g;
        pf.b += weight * (float) p.b;
        normalization += weight;
      }
      pf.r /= normalization;
      pf.g /= normalization;
      pf.b /= normalization;
      result[j * dst->width + i].SetClamp(pf.r, pf.g, pf.b);
    }
  }
}

void MagnifyY(Image* dst, Image* src, int smethod) {
  float y0 = 0.f;
  float weight = 0.f;
  float width = (smethod == IMAGE_SAMPLING_HAT) ? 2.f : 4.f;
  float normalization = 0.f;
  PixelFloat pf;
  Pixel* result = dst->pixels;
  float s = (float) dst->height / (float) src->height;

  for(int j = 0; j < dst->height; ++j) {
    for(int i = 0; i < dst->width; ++i) { 
      if(smethod == IMAGE_SAMPLING_POINT) {
        result[j * dst->width + i] = src->GetValidPixel(i, (float) j / ((float) dst->height - 1.f) * src->height);
        continue;
      }
      memset((void*) &pf.r, 0, sizeof(pf));
      normalization = 0.f;
      y0 = j / s; 
      for(int y = y0 - width; y <= y0 + width; ++y) {
        if(smethod == IMAGE_SAMPLING_HAT) {
          weight = HatWeight( y - j / s);
        } else if(smethod == IMAGE_SAMPLING_MITCHELL) {
          weight = MitchellWeight(y - j / s);
        }
        Pixel& p = src->GetValidPixel(i, y);
        pf.r += weight * (float) p.r;
        pf.g += weight * (float) p.g;
        pf.b += weight * (float) p.b;
        normalization += weight;
      }
      pf.r /= normalization;
      pf.g /= normalization;
      pf.b /= normalization;
      result[j * dst->width + i].SetClamp(pf.r, pf.g, pf.b);
    }
  }
}

void MinifyY(Image* dst, Image* src, int smethod) {
  float y0 = 0.f;
  float weight = 0.f;
  float normalization = 0.f;
  PixelFloat pf;
  Pixel* result = dst->pixels;
  float s = (float) dst->height / (float) src->height;
  float width = (smethod == IMAGE_SAMPLING_HAT) ? 2.f / s: 4.f / s;

  for(int j = 0; j < dst->height; ++j) {
    for(int i = 0; i < dst->width; ++i) { 
      if(smethod == IMAGE_SAMPLING_POINT) {
        result[j * dst->width + i] = src->GetValidPixel(i, (float) j / ((float) dst->height - 1.f) * src->height);
        continue;
      }
      memset((void*) &pf.r, 0, sizeof(pf));
      normalization = 0.f;
      y0 = j / s; 
      for(int y = y0 - width; y <= y0 + width; ++y) {
        if(smethod == IMAGE_SAMPLING_HAT) {
          weight = HatWeight(y * s - j); 
        } else if(smethod == IMAGE_SAMPLING_MITCHELL) {
          weight = MitchellWeight(y * s - j);
        }
        Pixel& p = src->GetValidPixel(i, y);
        pf.r += weight * (float) p.r;
        pf.g += weight * (float) p.g;
        pf.b += weight * (float) p.b;
        normalization += weight;
      }
      pf.r /= normalization;
      pf.g /= normalization;
      pf.b /= normalization;
      result[j * dst->width + i].SetClamp(pf.r, pf.g, pf.b);
    }
  }
}


Image* Image::Scale(int sizex, int sizey)
{
  if(sizex <= 0) {
    std::cerr << "Error. X dimension to scale must be greater than 0." << std::endl;
    return NULL;
  }

  if(sizey <= 0) {
    std::cerr << "Error. Y dimension to scale must be greater than 0." << std::endl;
    return NULL;
  }
  if(sampling_method != IMAGE_SAMPLING_POINT && 
     sampling_method != IMAGE_SAMPLING_HAT &&
     sampling_method != IMAGE_SAMPLING_MITCHELL) {
    std::cerr << "Error. Invalid image sampling type to Scale." << std::endl;
    return NULL;
  }

  float sx = (float) sizex / (float) width;
  float sy = (float) sizey / (float) height;
  Image* tempImage = new Image(sizex, height);

  if(sx > 1.f) {
    MagnifyX(tempImage, this, sampling_method);
  } else {
    MinifyX(tempImage, this, sampling_method);
  }

  Image* resultImage = new Image(sizex, sizey); 

  if(sy > 1.f) {
    MagnifyY(resultImage, tempImage, sampling_method);
  } else {
    MinifyY(resultImage, tempImage, sampling_method);
  }

  delete tempImage;
  return resultImage;
}

void ShiftX(Image* dst, Image* src, int smethod, float sx) {
  assert(src);
  assert(dst);
  float x0 = 0.f;
  float weight = 0.f;
  float normalization = 0.f;
  Pixel* result = dst->pixels;
  float width = (smethod == IMAGE_SAMPLING_HAT) ? 2.f : 4.f;
  PixelFloat pf;

  for(int j = 0; j < dst->height; ++j) {
    for(int i = 0; i < dst->width; ++i) {
      if(smethod == IMAGE_SAMPLING_POINT) {
        result[j * dst->width + i] = src->GetValidPixel(i - sx, j);
        continue;
      }
      memset((void*) &pf.r, 0, sizeof(pf));
      normalization = 0.f;
      x0 = (float) i - sx;
      for(int x = x0 - width; x <= x0 + width; ++x) {
        if(smethod == IMAGE_SAMPLING_HAT) {
          weight = HatWeight(x - i + sx);
        } else if(smethod == IMAGE_SAMPLING_MITCHELL) {
          weight = MitchellWeight(x - i + sx);
        }
        Pixel & p = src->GetValidPixel(x, j);
        pf.r += weight * (float) p.r;
        pf.g += weight * (float) p.g;
        pf.b += weight * (float) p.b;
        normalization += weight;
      }
      pf.r /= normalization;
      pf.g /= normalization;
      pf.b /= normalization;
      result[j * dst->width + i].SetClamp(pf.r, pf.g, pf.b);
    }
  }
}

void ShiftY(Image* dst, Image* src, float smethod, float sy) {
  assert(src);
  assert(dst);
  float y0 = 0.f;
  float weight = 0.f;
  float normalization = 0.f;

  Pixel* result = dst->pixels;
  float width = (smethod == IMAGE_SAMPLING_HAT) ? 2.f : 4.f;

  PixelFloat pf;

  for(int j = 0; j < dst->height; ++j) {
    for(int i = 0; i < dst->width; ++i) { 
      if(smethod == IMAGE_SAMPLING_POINT) {
        result[j * dst->width + i] = src->GetValidPixel(i, j - sy);
        continue;
      }
      memset((void*) &pf.r, 0, sizeof(pf));
      normalization = 0.f;
      y0 = j - sy;
      for(int y = y0 - width; y <= y0 + width; ++y) {
        if(smethod == IMAGE_SAMPLING_HAT) {
          weight = HatWeight(y - j +  sy); 
        } else if(smethod == IMAGE_SAMPLING_MITCHELL) {
          weight = MitchellWeight(y - j + sy);
        }
        Pixel& p = src->GetValidPixel(i, y);
        pf.r += weight * (float) p.r;
        pf.g += weight * (float) p.g;
        pf.b += weight * (float) p.b;
        normalization += weight;
      }
      pf.r /= normalization;
      pf.g /= normalization;
      pf.b /= normalization;
      result[j * dst->width + i].SetClamp(pf.r, pf.g, pf.b);

    }
  }

}

void Image::Shift(double sx, double sy)
{
  if(sampling_method != IMAGE_SAMPLING_POINT && 
     sampling_method != IMAGE_SAMPLING_HAT &&
     sampling_method != IMAGE_SAMPLING_MITCHELL) {
    std::cerr << "Error. Invalid image sampling type to Shift." << std::endl;
    return;
  }
  Image* tempImage = new Image(width, height);
  Pixel* tempPixels = tempImage->pixels;
  for(int i = 0; i < num_pixels; ++i) {
    tempPixels[i].r = 0;
    tempPixels[i].g = 0;
    tempPixels[i].b = 0;
  }
  ShiftX(tempImage, this, sampling_method, sx);
  ShiftY(this, tempImage, sampling_method, sy);
  delete tempImage;
}


/*
Image* Image::Rotate(double angle)
{
  // For rotation of the image
  // Not required in the assignment
  // But you can earn limited extra credit if you fill it in
  // (It isn't really that hard) 

    return NULL;
}
*/


void Image::Fun()
{
    /* Your Work Here (Section 3.6) */
}


Image* ImageMorph (Image* I0, Image* I1, int numLines, Line* L0, Line* L1, double t)
{
  /* Your Work Here (Section 3.7) */
  // This is extra credit.
  // You can modify the function definition. 
  // This definition takes two images I0 and I1, the number of lines for 
  // morphing, and a definition of corresponding line segments L0 and L1
  // t is a parameter ranging from 0 to 1.
  // For full credit, you must write a user interface to join corresponding 
  // lines.
  // As well as prepare movies 
  // An interactive slider to look at various morph positions would be good.
  // From Beier-Neely's SIGGRAPH 92 paper

    return NULL;
}


/**
 * Image Sample
 **/
void Image::SetSamplingMethod(int method)
{
  // Sets the filter to use for Scale and Shift
  // You need to implement point sampling, hat filter and mitchell

    assert((method >= 0) && (method < IMAGE_N_SAMPLING_METHODS));
    sampling_method = method;
}

Pixel Image::Sample (double u, double v, double sx, double sy)
{
  // To sample the image in scale and shift
  // This is an auxiliary function that it is not essential you fill in or 
  // you may define it differently.
  // u and v are the floating point coords of the points to be sampled.
  // sx and sy correspond to the scale values. 
  // In the assignment, it says implement MinifyX MinifyY MagnifyX MagnifyY
  // separately.  That may be a better way to do it.
  // This hook is primarily to get you thinking about that you have to have 
  // some equivalent of this function.

  if (sampling_method == IMAGE_SAMPLING_POINT) {
    // Your work here
  }

  else if (sampling_method == IMAGE_SAMPLING_HAT) {
    // Your work here
  }

  else if (sampling_method == IMAGE_SAMPLING_MITCHELL) {
    // Your work here
  }

  else {
    fprintf(stderr,"I don't understand what sampling method is used\n") ;
    exit(1) ;
  }

  return Pixel() ;
}

