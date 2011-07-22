itk_module_test()

set(ITK-AffineImageTests
itkAffineImageTest.cxx
)

CreateTestDriver(ITK-AffineImage "${ITK-AffineImage-Test_LIBRARIES}" "${ITK-AffineImageTests}")

itk_add_test( NAME itkAffineImageTest
  COMMAND ITK-AffineImageTestDriver itkAffineImageTest )
