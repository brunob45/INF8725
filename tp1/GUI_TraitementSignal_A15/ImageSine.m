function ImageSignal = ImageSine( Amplitude, PeriodH, PeriodW, PhaseH, PhaseW, ImageSize )
% Create an image containing a single waveform that is configurable.
%
% Inputs:
%   Angle : The angle of the waveform, in radian
%   Frequency : The frequency of the waveform
%   ImageSize : Size of the image. The output image is square.
% Outputs:
%   ImageSignal : The resulting image of the waveform.

    Tw = PeriodW;
    Tw = Clamp( Tw, 1, ImageSize );
    
    Th = PeriodH;
    Th = Clamp( Th, 1, ImageSize );
    
    PhiW = PhaseW;
    PhiW = Clamp( PhiW, 0, 2*pi );
    
    PhiH = PhaseH;
    PhiH = Clamp( PhiH, 0, 2*pi );
    
    A = Amplitude;
    
    [X Y] = meshgrid( 1:ImageSize, 1:ImageSize );
    ImageSignal = A * sin( (2 * pi / Tw) * X + PhiW + (2 * pi / Th) * Y + PhiH );
    
    MinVal = min( ImageSignal(:) );
    MaxVal = max( ImageSignal(:) );
    Range = MaxVal - MinVal;
    ImageSignal = (ImageSignal - MinVal) / Range * 255;

end

function ClampedVal = Clamp( Value, MinVal, MaxVal )
    ClampedVal = min( max( Value, MinVal ), MaxVal );
end