/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include "itkDemonsRegistrationFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkIndex.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkWarpImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkCommand.h"
#include "vnl/vnl_math.h"
#include "itkVectorCastImageFilter.h"
#include "itkAffineTransform.h"
#include "itkResampleImageFilter.h"
namespace{
// The following class is used to support callbacks
// on the filter in the pipeline that follows later

// Template function to fill in an image with a circle.
template <class TImage>
void
FillWithEllipse(
TImage * image,
double * center,
double radius, double c, 
typename TImage::PixelType foregnd,
typename TImage::PixelType backgnd)
{

  typedef itk::ImageRegionIteratorWithIndex<TImage> Iterator;
  Iterator it( image, image->GetBufferedRegion() );
  it.Begin();

  typename TImage::IndexType index;
  for( ; !it.IsAtEnd(); ++it )
    {
    index = it.GetIndex();
    double distancea = sqrt( vnl_math_sqr((double) index[0] - (double) center[0] + c )
			   + vnl_math_sqr((double) index[1]  - center[1]) );
    double distanceb = sqrt( vnl_math_sqr((double) index[0] - (double) center[0] - c )
			   + vnl_math_sqr((double) index[1]  - center[1]) );
    double distance = distancea + distanceb;
    if( distance <= 2*radius ) it.Set( foregnd );
    else it.Set( backgnd );
    }

}


}

int itkAffineImageTest(int, char* [] )
{

  typedef float PixelType;
  enum {ImageDimension = 2};
  typedef itk::Image<PixelType,ImageDimension> ImageType;
  typedef itk::Vector<float,ImageDimension> VectorType;
  typedef itk::Image<VectorType,ImageDimension> FieldType;
  typedef itk::Image<VectorType::ValueType,ImageDimension> FloatImageType;
  typedef ImageType::IndexType  IndexType;
  typedef ImageType::SizeType   SizeType;
  typedef ImageType::RegionType RegionType;

  //--------------------------------------------------------
  std::cout << "Generate input images and initial deformation field";
  std::cout << std::endl;

  ImageType::SizeValueType sizeArray[ImageDimension] = { 128, 128 };
  SizeType size;
  size.SetSize( sizeArray );

  IndexType index;
  index.Fill( 0 );

  RegionType region;
  region.SetSize( size );
  region.SetIndex( index );

  ImageType::Pointer fixed_resampled;
  ImageType::Pointer fixed = ImageType::New();

  fixed->SetLargestPossibleRegion( region );
  fixed->SetBufferedRegion( region );
  fixed->Allocate();

  double center[ImageDimension];
  double radius;
  PixelType fgnd = 1;
  PixelType bgnd = 0;

  // fill fixed with circle
  center[0] = 64; center[1] = 64; radius = 30;
  FillWithEllipse<ImageType>( fixed, center, radius, 0, fgnd, bgnd );
  /** define a non-identity direction matrix in the fixed image */
  ImageType::DirectionType direction=fixed->GetDirection( );
  direction[0][0]=-1;
  direction[1][0]=0.05;
  fixed->SetDirection(direction);
  double transx=30;
  ImageType::PointType ftranslation;  ftranslation.Fill(0); ftranslation[1]=transx;
  fixed->SetOrigin(ftranslation);

  /** transform the fixed by a known affine transform */  
  typedef itk::AffineTransform< double, ImageDimension > TransformType;
  TransformType::Pointer      transform     = TransformType::New();
  transform->SetIdentity();
  TransformType::ParametersType params=transform->GetParameters();
  params[0]=1.1;
  params[1]=0.1;
  double transy=20;
  params[5]=transy;
  transform->SetParameters(params);
  typedef itk::ResampleImageFilter< ImageType, ImageType > ResamplerType;
  ResamplerType::Pointer resampler = ResamplerType::New();
  resampler->SetInput( fixed );
  resampler->SetTransform( transform );
  resampler->SetSize( fixed->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin(fixed->GetOrigin() );
  resampler->SetOutputSpacing(fixed->GetSpacing() );
  resampler->SetOutputDirection(fixed->GetDirection());
  resampler->SetDefaultPixelValue( 0 );
  resampler->Update();
  fixed_resampled=resampler->GetOutput();
  /** At this point, fixed_resampled is no longer aligned with fixed in physical space. */

  /* But now, we put the affine transform back into the resampled image... */
  fixed_resampled->SetDirection( transform->GetMatrix() * fixed->GetDirection() );
  ImageType::PointType translation;  
  translation.Fill(0); 
  translation[1]=transy;
  itk::Vector<double, 2> fixed_origin;
  for (unsigned int i=0; i<ImageDimension; i++) fixed_origin[i]=fixed->GetOrigin()[i];
  itk::Vector<double, 2> transformed_fixed_origin=transform->GetMatrix() * fixed_origin;
  for (unsigned int i=0; i<ImageDimension; i++) 
    translation[i]=transformed_fixed_origin[i]+translation[i];
  fixed_resampled->SetOrigin(translation);

  /** open the images written out below in ITK-SNAP to see that they are aligned, but in different physical space*/
  typedef itk::ImageFileWriter< ImageType >  WriterType;
  WriterType::Pointer      writer =  WriterType::New();
  writer->SetFileName( "zfixed.mhd" );
  writer->SetInput( fixed );
  writer->Update();
  WriterType::Pointer      writer2 =  WriterType::New();
  writer2->SetFileName( "zfixed_resampled.mhd" );
  writer2->SetInput( fixed_resampled );
  writer2->Update();

  std::cout << "Compare warped moving and fixed in index space and physical space. " << std::endl;

  // compare the warp and fixed images
  itk::ImageRegionIteratorWithIndex<ImageType> fixedIter( fixed,
      fixed->GetBufferedRegion() );
  typedef itk:: NearestNeighborInterpolateImageFunction<
                                    ImageType,
                                    double >             InterpolatorType;
  InterpolatorType::Pointer   interpolator  = InterpolatorType::New();
  interpolator->SetInputImage( fixed_resampled );

  double error_index_space=0;
  double error_physical_space=0;
  unsigned int ct=0;
  fixedIter.GoToBegin();
  while( !fixedIter.IsAtEnd() )
    {
      error_index_space+=fabs(fixedIter.Get()-fixed_resampled->GetPixel(fixedIter.GetIndex()));
      ImageType::PointType point;
      fixed->TransformIndexToPhysicalPoint(fixedIter.GetIndex(), point);

      // now do physical space
      if ( interpolator->IsInsideBuffer(point) )
      {
        double value = interpolator->Evaluate(point);
        error_physical_space+=fabs(fixedIter.Get()-value);
      }
      // Check boundaries and assign
      if ( fixedIter.Get() > 0 ) ct++;
      ++fixedIter;
    }

  std::cout << "Average difference in index space : " <<error_index_space/(float)ct <<std::endl;
  std::cout << "Average difference in phys space : " <<error_physical_space/(float)ct ;
  std::cout << std::endl;

  std::cout <<" write a z*** nifti image for comparison " << std::endl;
  WriterType::Pointer      writer3 =  WriterType::New();
  writer3->SetFileName( "zfixed_resampled.nii.gz" );
  writer3->SetInput( fixed_resampled );
  writer3->Update();

  std::cout <<" write a z*** nrrd image for comparison " << std::endl;
  WriterType::Pointer      writer4 =  WriterType::New();
  writer4->SetFileName( "zfixed_resampled.nrrd" );
  writer4->SetInput( fixed_resampled );
  writer4->Update();

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer nrrdReader = ReaderType::New();
  nrrdReader->SetFileName("zfixed_resampled.nrrd");
  nrrdReader->Update();
  ReaderType::Pointer niftiReader = ReaderType::New();
  niftiReader->SetFileName("zfixed_resampled.nii.gz");
  niftiReader->Update();
  ReaderType::Pointer mhdReader = ReaderType::New();
  mhdReader->SetFileName("zfixed_resampled.mhd");
  mhdReader->Update();

  double merror_physical_space=0;
  double nrerror_physical_space=0;
  double nferror_physical_space=0;
  fixedIter.GoToBegin();
  while( !fixedIter.IsAtEnd() )
    {
      ImageType::PointType point;
      fixed->TransformIndexToPhysicalPoint(fixedIter.GetIndex(), point);

      interpolator->SetInputImage( nrrdReader->GetOutput() );
      if ( interpolator->IsInsideBuffer(point) )
      {
        double value = interpolator->Evaluate(point);
        nrerror_physical_space+=fabs(fixedIter.Get()-value);
      }

      interpolator->SetInputImage( niftiReader->GetOutput() );
      if ( interpolator->IsInsideBuffer(point) )
      {
        double value = interpolator->Evaluate(point);
        nferror_physical_space+=fabs(fixedIter.Get()-value);
      }

      interpolator->SetInputImage( mhdReader->GetOutput() );
      if ( interpolator->IsInsideBuffer(point) )
      {
        double value = interpolator->Evaluate(point);
        merror_physical_space+=fabs(fixedIter.Get()-value);
      }
      ++fixedIter;
    }

  std::cout << "Average difference in index space : " <<error_index_space/(float)ct << std::endl;
  std::cout << "Average difference in phys space mhd : " <<merror_physical_space/(float)ct  << std::endl;
  std::cout << "Average difference in phys space nrrd : " <<nrerror_physical_space/(float)ct  << std::endl;
  std::cout << "Average difference in phys space nii : " <<nferror_physical_space/(float)ct  << std::endl;
  std::cout << std::endl;

  if ( error_physical_space/(float)ct > 0.05 ) return EXIT_FAILURE;
  std::cout << "Test passed" << std::endl;
  return EXIT_SUCCESS;

}

