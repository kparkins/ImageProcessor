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
    std::cerr << "w greater " << w << std::endl;
    w = width - x;
  }

  if(y + h > height) {
    std::cerr << "h greater " << h << std::endl;
    h = height - y;
  }

  int dx = 0;
  int dy = 0;
  Image* image = new Image(w, h);
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

void Image::Quantize (int nbits)
{
  /* Your Work Here (Section 3.3.1) */
}


void Image::RandomDither (int nbits)
{
  /* Your Work Here (Section 3.3.2) */
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

/* Error-diffusion parameters for Floyd-Steinberg*/
const double
    ALPHA = 7.0 / 16.0,
    BETA  = 3.0 / 16.0,
    GAMMA = 5.0 / 16.0,
    DELTA = 1.0 / 16.0;

void Image::FloydSteinbergDither(int nbits)
{
  /* Your Work Here (Section 3.3.3) */
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
  
  // It is helpful if you write an auxiliary convolve function.
  // But this form is just for guidance and is completely optional.
  // Your solution NEED NOT fill in this function at all
  // Or it can use an alternate form or definition
}

void Image::Blur(int n)
{
  /* Your Work Here (Section 3.4.1) */
}

void Image::Sharpen() 
{
  /* Your Work Here (Section 3.4.2) */
}

void Image::EdgeDetect(int threshold)
{
  /* Your Work Here (Section 3.4.3) */
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

