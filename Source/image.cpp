#include "image.h"
#include "bmp.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>


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

  for(int i = 0; i < num_pixels; ++i) {
    pixels[i] = pixels[i] * factor;
    pixels[i].SetClamp(pixels[i].r, pixels[i].g, pixels[i].b);
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

  Pixel averagePixel(luminance, luminance, luminance, 1.f);
  for(int i = 0; i < num_pixels; ++i) {
    if(invert) {
      pixels[i].Set(255 - pixels[i].r, 255 - pixels[i].g, 255 - pixels[i].b);
    }
    pixels[i] =  averagePixel * (1 - factor) + pixels[i] * factor;
    pixels[i].SetClamp(pixels[i].r, pixels[i].g, pixels[i].b);
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
    float greyColor = (pixels[i].r + pixels[i].g + pixels[i].b) / 3;
    Pixel greyPixel(greyColor, greyColor, greyColor);

    if(invert) {
      pixels[i].Set(255 - pixels[i].r, 255 - pixels[i].g, 255 - pixels[i].b);
    }

    pixels[i] = greyPixel * (1 - factor) + pixels[i] * factor;
    pixels[i].SetClamp(pixels[i].r, pixels[i].g, pixels[i].b);
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
    pixels[i].r = pow(pixels[i].r, 1.f / factor);   
    pixels[i].g = pow(pixels[i].g, 1.f / factor);   
    pixels[i].b = pow(pixels[i].b, 1.f / factor);   
  }
}

Image* Image::Crop(int x, int y, int w, int h)
{
  if(x < 0 || y < 0 || w < 0 || h < 0) {
    std::cerr << "Error. Negative argument to Image::Crop." << std::endl;
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

  int dx = 0;
  int dy = 0;
  Image* image = new Image(w, h);
  assert(image != NULL);
  Pixel* targetPixels = image->pixels;
  for(int sy = y; sy < height; ++sy, ++dy) {
    for(int sx = x; sx < width; ++sx, ++dx) {
      targetPixels[dy * w + dx] = pixels[sy * width + sx];
    }
    dx = 0;
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
  if(quantum < 0) {
    quantum = 0;
  } else if(quantum > numChannels - 1) {
    quantum = numChannels - 1;
  }
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


Pixel& Image::GetValidPixel(int x, int y) {
  return pixels[mod<int>(y, height) * width + mod<int>(x, width)];
}

void Image::UpdatePixelError(int x, int y, float r, float g, float b, double weight) {
  Pixel & p = GetValidPixel(x,y);
  p.r = clamp(0.f, 255.f, (float) p.r + r * weight);
  p.g = clamp(0.f, 255.f, (float) p.g + g * weight);
  p.b = clamp(0.f, 255.f, (float) p.b + b * weight);
}

void Image::FloydSteinbergDither(int nbits)
{
  if(nbits < 1 || nbits > 8) {
    std::cerr << "Error. Random Dither only accepts arguments in the range [1, 8]." << std::endl;
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

void Image::Convolve(int *filter, int n, int normalization, int absval) {
  // This is my definition of an auxiliary function for image convolution 
  // with an integer filter of width n and certain normalization.
  // The absval param is to consider absolute values for edge detection.
 
  int d = (n - 1) / 2;
  float r = 0;
  float g = 0;
  float b = 0;
  float weight = 0;

  std::cerr << normalization << std::endl;
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
  
  this->Convolve(kernel, n, normalization, 0);
}

void Image::Sharpen() 
{
  static int[9] kernel = {
    -1, -2, -1, 
    -2, 19, -2, 
    -1, -2, -1
  }
  this->Convolve(kernel, 3, 7, 0);
}

void Image::EdgeDetect(int threshold)
{
}


Image* Image::Scale(int sizex, int sizey)
{
  /* Your Work Here (Section 3.5.1) */
  return NULL ;
}

void Image::Shift(double sx, double sy)
{
  /* Your Work Here (Section 3.5.2) */
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

