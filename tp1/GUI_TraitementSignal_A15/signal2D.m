
function varargout = signal2D(varargin)
% SIGNAL2D M-file for signal2D.fig
%      SIGNAL2D, by itself, creates a new SIGNAL2D or raises the existing
%      singleton*.
%
%      H = SIGNAL2D returns the handle to a new SIGNAL2D or the handle to
%      the existing singleton*.
%
%      SIGNAL2D('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SIGNAL2D.M with the given input arguments.
%
%      SIGNAL2D('Property','Value',...) creates a new SIGNAL2D or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before signal2D_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to signal2D_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help signal2D

% Last Modified by GUIDE v2.5 08-Jun-2015 16:04:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @signal2D_OpeningFcn, ...
    'gui_OutputFcn',  @signal2D_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before signal2D is made visible.
function signal2D_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to signal2D (see VARARGIN)

% Choose default command line output for signal2D
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes signal2D wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% on ajoute le path pour que l interface continue meme si on change le
% dossier
p=mfilename('fullpath');
[pathstr, name, ext] = fileparts(p);

set(findobj('Tag','tb1_open'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb1_save'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb2_open'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb2_save'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb2_in_t_zoom'),'CData',iconRead(fullfile(pathstr, 'zoom.mat')));
set(findobj('Tag','tb2_out_t_zoom'),'CData',iconRead(fullfile(pathstr, 'zoom.mat')));
set(findobj('Tag','tb2_filtre_zoom'),'CData',iconRead(fullfile(pathstr, 'rotate.mat')));
set(findobj('Tag','tb3_in_d_z'),'CData',iconRead(fullfile(pathstr, 'zoom.mat')));
set(findobj('Tag','tb3_out_z'),'CData',iconRead(fullfile(pathstr, 'zoom.mat')));
set(findobj('Tag','tb3_open'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb3_save'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb4_open'),'CData',iconRead(fullfile(pathstr, 'opendoc.mat')));
set(findobj('Tag','tb4_save'),'CData',iconRead(fullfile(pathstr, 'savedoc.mat')));
set(findobj('Tag','tb4_in_z'),'CData',iconRead(fullfile(pathstr, 'zoom.mat')));
set(findobj('Tag','tb4_out_z'),'CData',iconRead(fullfile(pathstr, 'zoom.mat')));
tb1_clear_axes(hObject, eventdata, handles);
tb2_clear_axes(hObject, eventdata, handles);

addpath( fullfile(pathstr,'export_fig' ));

% on ajoute le path pour que l interface continue meme si on change le
% dossier
p=mfilename('fullpath');
[pathstr, name, ext] = fileparts(p);
addpath( pathstr );

% --- Outputs from this function are returned to the command line.
function varargout = signal2D_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in tb1_pop.
function tb1_pop_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_pop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb1_pop contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb1_pop
axes(handles.tb1_out);
cla;

switch get(findobj('Tag','tb1_pop'),'Value')
    case 1
        set(findobj('Tag','tb1_N'),'Visible','off');
        set(findobj('Tag','tb1_texttaille'),'Visible','off');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 2
        set(findobj('Tag','tb1_N'),'Visible','on');
        set(findobj('Tag','tb1_texttaille'),'Visible','on');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 3
        set(findobj('Tag','tb1_N'),'Visible','on');
        set(findobj('Tag','tb1_texttaille'),'Visible','on');
        set(findobj('Tag','tb1_sigma'),'Visible','on');
        set(findobj('Tag','tb1_textecart'),'Visible','on');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 4
        set(findobj('Tag','tb1_N'),'Visible','on');
        set(findobj('Tag','tb1_texttaille'),'Visible','on');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 5
        set(findobj('Tag','tb1_N'),'Visible','on');
        set(findobj('Tag','tb1_texttaille'),'Visible','on');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 6
        set(findobj('Tag','tb1_N'),'Visible','off');
        set(findobj('Tag','tb1_texttaille'),'Visible','off');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','on');
        set(findobj('Tag','tb1_prewitt'),'Visible','on');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 7
        set(findobj('Tag','tb1_N'),'Visible','off');
        set(findobj('Tag','tb1_texttaille'),'Visible','off');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 8
        set(findobj('Tag','tb1_N'),'Visible','on');
        set(findobj('Tag','tb1_texttaille'),'Visible','on');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','on');
        set(findobj('Tag','tb1_strelpop'),'Visible','on');
        set(findobj('Tag','tb1_textstrel'),'Visible','on');
        set(findobj('Tag','tb1_popbruit'),'Visible','off');
    case 9
        set(findobj('Tag','tb1_popbruit'),'Visible','on');
        set(findobj('Tag','tb1_N'),'Visible','on');
        set(findobj('Tag','tb1_texttaille'),'Visible','on');
        set(findobj('Tag','tb1_sigma'),'Visible','off');
        set(findobj('Tag','tb1_textecart'),'Visible','off');
        set(findobj('Tag','tb1_sobel'),'Visible','off');
        set(findobj('Tag','tb1_prewitt'),'Visible','off');
        set(findobj('Tag','tb1_morphopop'),'Visible','off');
        set(findobj('Tag','tb1_strelpop'),'Visible','off');
        set(findobj('Tag','tb1_textstrel'),'Visible','off');
    otherwise
end

% --- Executes on selection change in tb1_popbruit.
function tb1_popbruit_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_popbruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb1_out);
cla;
switch get(findobj('Tag','tb1_popbruit'),'Value')
    case 1
        set(findobj('Tag','tb1_textordre'),'Visible','off');
        set(findobj('Tag','tb1_ordre'),'Visible','off');
    case 2
        set(findobj('Tag','tb1_textordre'),'Visible','off');
        set(findobj('Tag','tb1_ordre'),'Visible','off');
    case 3
        set(findobj('Tag','tb1_textordre'),'Visible','on');
        set(findobj('Tag','tb1_ordre'),'Visible','on');
    case 4
        set(findobj('Tag','tb1_textordre'),'Visible','off');
        set(findobj('Tag','tb1_ordre'),'Visible','off');
    case 5
        set(findobj('Tag','tb1_textordre'),'Visible','off');
        set(findobj('Tag','tb1_ordre'),'Visible','off');
    case 6
        set(findobj('Tag','tb1_textordre'),'Visible','off');
        set(findobj('Tag','tb1_ordre'),'Visible','off');
    case 7
        set(findobj('Tag','tb1_textordre'),'Visible','off');
        set(findobj('Tag','tb1_ordre'),'Visible','off');
    otherwise
end


% --- Executes during object creation, after setting all properties.
function tb1_popbruit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_popbruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function tb1_pop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_pop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function tb1_ordre_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_ordre (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb1_ordre as text
%        str2double(get(hObject,'String')) returns contents of tb1_ordre as a double


% --- Executes during object creation, after setting all properties.
function tb1_ordre_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_ordre (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb1_density_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_density (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb1_density as text
%        str2double(get(hObject,'String')) returns contents of tb1_density as a double


% --- Executes during object creation, after setting all properties.
function tb1_density_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_density (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb1_pbaddnoise.
function tb1_pbaddnoise_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_pbaddnoise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h=findobj('Tag','tb2_filtre_f');
tb1_in=getimage(h);

d=get(findobj('Tag','tb1_slider_bruit'),'Value');
switch get(findobj('Tag','tb1_popaddnoise'),'Value')
    case 1
        tb1_in_n=imnoise(tb1_in,'gaussian',0,0.05*d);
    case 2
        tb1_in_n=imnoise(tb1_in,'salt & pepper',d);
    otherwise
end

axes(handles.tb1_in);
h=imshow(tb1_in_n);
set(h,'Tag','tb1_in');



% --- Executes on selection change in tb1_popaddnoise.
function tb1_popaddnoise_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_popaddnoise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb1_popaddnoise contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb1_popaddnoise


% --- Executes during object creation, after setting all properties.
function tb1_popaddnoise_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_popaddnoise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in tb1_filtrer.
function tb1_filtrer_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_filtrer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


h=findobj('Tag','tb1_in');
tb1_in=getimage(h);

switch get(findobj('Tag','tb1_pop'),'Value')
    case 1
        %egalisation histogramme
        axes(handles.tb1_out);
        tb1_out=histeq(tb1_in,256);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
        
        
        figure, imhist(tb1_in,256);title('Histogramme image originale')
        figure, imhist(tb1_out,256);title('Histogramme image rehaussée')
    case 2
        % filtre moyen
        N=str2num(get(findobj('Tag','tb1_N'),'String'));
        mask=fspecial('average',N);
        tb1_out=filter2(mask,tb1_in);
        
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
    case 3
        % filtre Gaussien
        N=str2num(get(findobj('Tag','tb1_N'),'String'));
        sigma=str2num(get(findobj('Tag','tb1_sigma'),'String'));
        mask=fspecial('gaussian',N,sigma);
        
        tb1_out=imfilter(tb1_in,mask);
        
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
        figure,mesh(mask);title('masque Gaussien')
    case 4
        % filtre median
        N=str2num(get(findobj('Tag','tb1_N'),'String'));
        tb1_out=medfilt2(tb1_in,[N N]);
        
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
    case 5
        % filtre median adaptatif
        N=str2num(get(findobj('Tag','tb1_N'),'String'));
        tb1_out=medfilt2_a(tb1_in,N);
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
    case 6
        % gradient
        switch get(findobj('Tag','tb1_prewitt'),'Value')
            case 1
                [tb1_out dir] = imgradient(tb1_in, 'Sobel');
            case 0
                [tb1_out dir] = imgradient(tb1_in, 'Prewitt');
        end
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
    case 7
        %Laplacien
        
        mask=fspecial('laplacian');
        tb1_out=imfilter((tb1_in),mask);
        
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
    case 8
        %morpho
        shape={'square','disk','diamond'};
        N=str2num(get(findobj('Tag','tb1_N'),'String'));
        ishape=get(findobj('Tag','tb1_strelpop'),'Value');
        se=strel(shape{ishape},N);
        
        switch get(findobj('Tag','tb1_morphopop'),'Value')
            case 1
                %erosion
                tb1_out=imerode(tb1_in,se);
            case 2
                %originale moins erosion
                tb1_out=tb1_in-imerode(tb1_in,se);
            case 3
                %dilatation
                tb1_out=imdilate(tb1_in,se);
            case 4
                %ouverture
                tb1_out=imopen(tb1_in,se);
            case 5
                %fermeture
                tb1_out=imclose(tb1_in,se);
            case 6
                %ouverture+fermeture
                tb1_out=imclose(imopen(tb1_in,se),se);
            otherwise
        end
        
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
    case 9
        T=str2num(get(findobj('Tag','tb1_N'),'String'));
        Tm=floor(T/2);
        [n m]=size(tb1_in);
        tb1_out=zeros(n,m);
        switch get(findobj('Tag','tb1_popbruit'),'Value')
            case 1
                % filtre geometrique
                for i=1:n
                    for j=1:m
                        S=tb1_in(max([1 i-Tm]):min([n i+Tm]),max([1 j-Tm]):min([m j+Tm]));
                        tb1_out(i,j)=prod(S(:)).^(1/(T*T));
                    end
                end
            case 2
                % filtre harmonique
                for i=1:n
                    for j=1:m
                        S=tb1_in(max([1 i-Tm]):min([n i+Tm]),max([1 j-Tm]):min([m j+Tm]));
                        tb1_out(i,j)=T*T./sum(1./S(:));
                    end
                end
            case 3
                % filtre contraharonique
                Q=str2num(get(findobj('Tag','tb1_ordre'),'String'));
                for i=1:n
                    for j=1:m
                        S=tb1_in(max([1 i-Tm]):min([n i+Tm]),max([1 j-Tm]):min([m j+Tm]));
                        tb1_out(i,j)=sum(S(:).^(Q+1))/sum(S(:).^(Q));
                    end
                end
            case 4
                % filtre max
                for i=1:n
                    for j=1:m
                        S=tb1_in(max([1 i-Tm]):min([n i+Tm]),max([1 j-Tm]):min([m j+Tm]));
                        tb1_out(i,j)=max(S(:));
                    end
                end
            case 5
                % filtre min
                for i=1:n
                    for j=1:m
                        S=tb1_in(max([1 i-Tm]):min([n i+Tm]),max([1 j-Tm]):min([m j+Tm]));
                        tb1_out(i,j)=min(S(:));
                    end
                end
            case 6
                % filtre dichotomique
                for i=1:n
                    for j=1:m
                        S=tb1_in(max([1 i-Tm]):min([n i+Tm]),max([1 j-Tm]):min([m j+Tm]));
                        tb1_out(i,j)=0.5*(max(S(:))+min(S(:)));
                    end
                end
            otherwise
        end
        axes(handles.tb1_out);
        h=imshow((tb1_out));
        set(h,'Tag','tb1_out');
    otherwise
end


% median adaptatif
function [Icf]=medfilt2_a(I,s)

if s == 0
    Icf = I;
    return;
end

[n m]=size(I);

Ic=zeros(n+2*s,m+2*s);
Icf=zeros(n+2*s,m+2*s);
Ic(s+1:end-s,s+1:end-s)=I;

for i=s+1:n+s
    for j=s+1:m+s
        st=1;
        flag=0;
        while st<=floor(s/2) & flag==0
            V=Ic(i-st:i+st,j-st:j+st);
            zmed=median(V(:));
            zmax=max(max(V));
            zmin=min(min(V));
            A1=zmed-zmin;
            A2=zmed-zmax;
            if A1>0 & A2<0
                zxy=Ic(i,j);
                B1=zxy-zmin;
                B2=zxy-zmax;
                if B1>0 & B2<0
                    Icf(i,j)=zmed;
                    flag=1;
                else
                    Icf(i,j)=zmed;
                    flag=1;
                end
            else
                st=st+1;
            end
        end
        if st>floor(s/2) & flag==0
            Icf(i,j)=zmed;
        end
    end
end



% --- Executes on button press in tb1_open.
function tb1_open_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb1_out);
cla;

[FileName,PathName] = uigetfile('*.*','Choisir une image');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end

[in map]=imread([PathName FileName]);
if ~isempty(map)
    in=ind2gray(in, map);
end
if size(in,3)==3
    in=rgb2gray(in);
end
if islogical(in)
    in = cast(in, 'uint8');
    in = in .* 255;
end
in=double(in);
in=in./255;
axes(handles.tb1_in);
h=imshow(in);
set(h,'Tag','tb1_in');
axes(handles.tb2_in_t);
h=imshow(in);
set(h,'Tag','tb2_in_t');

h=findobj('Tag','tb2_in_t');
tb1_in=getimage(h(size(h,1)));

d=get(findobj('Tag','tb1_slider_bruit'),'Value');
switch get(findobj('Tag','tb1_popaddnoise'),'Value')
    case 1
        tb1_in_n=imnoise(tb1_in,'gaussian',0,0.2*d);
        set(findobj('Tag','tb1_text_sigma'),'String',['Bruit Gaussien de moyenne 0 et de sigma ' num2str(0.2*d)]);
    case 2
        tb1_in_n=imnoise(tb1_in,'salt & pepper',d);
        set(findobj('Tag','tb1_text_sigma'),'String',['Bruit Poivre et sel de densité ' num2str(d)]);
    otherwise
end

axes(handles.tb1_in);
h=imshow(tb1_in_n);
set(h,'Tag','tb1_in');
axes(handles.tb2_in_t);
h=imshow(tb1_in);
set(h,'Tag','tb2_in_t');

% --- Executes on button press in tb1_save.
function tb1_save_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h=findobj('Tag','tb1_out');
tb1_out=getimage(h);
if isempty(tb1_out)
    return;
end

[file,path] = uiputfile('*.jpg','Sauvegarder image filtree');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
imwrite(tb1_out,[path file]);



function tb1_N_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_N (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb1_N as text
%        str2double(get(hObject,'String')) returns contents of tb1_N as a double


% --- Executes during object creation, after setting all properties.
function tb1_N_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_N (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb1_sobel.
function tb1_sobel_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_sobel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb1_prewitt'),'Value',0);

% --- Executes on button press in tb1_prewitt.
function tb1_prewitt_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_prewitt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(findobj('Tag','tb1_sobel'),'Value',0);


% --- Executes on selection change in tb1_morphopop.
function tb1_morphopop_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_morphopop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb1_morphopop contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb1_morphopop


% --- Executes during object creation, after setting all properties.
function tb1_morphopop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_morphopop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in tb1_strelpop.
function tb1_strelpop_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_strelpop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb1_strelpop contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb1_strelpop


% --- Executes during object creation, after setting all properties.
function tb1_strelpop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_strelpop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function tb1_clear_axes(hObject, eventdata, handles)
axes(handles.tb1_in);
cla;
axes(handles.tb1_out);
cla;




function tb2_clear_axes(hObject, eventdata, handles)
axes(handles.tb2_in_t);
cla;
axes(handles.tb2_in_f);
cla;
axes(handles.tb2_out_t);
cla;
axes(handles.tb2_out_f);
cla;



% --- Executes on button press in tb2_open.
function tb2_open_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
tb2_clear_axes(hObject, eventdata, handles);

[FileName,PathName] = uigetfile('*.*','Choisir une image');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end
[in map]=imread([PathName FileName]);
if ~isempty(map)
    in=ind2gray(in, map);
end
if size(in,3)==3
    in=rgb2gray(in);
end
if islogical(in)
    in = cast(in, 'uint8');
    in = in .* 255;
end
in=double(in);
in=in./255;
axes(handles.tb2_in_t);
h=imshow(in);
set(h,'Tag','tb2_in_t');

%in_f=fftshift(fft2(in));

[n m] = size(in);
PadHeight = 2^nextpow2(n);
PadWidth = 2^nextpow2(m);
in_f=fftshift(fft2(in, PadHeight, PadWidth));
% in_f=255*in_f./max(max(in_f));
axes(handles.tb2_in_f);
h=imshow(log(1+abs(in_f)), []);
set(h,'Tag','tb2_in_f');


% --- Executes on button press in tb2_save.
function tb2_save_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb2_out_t');
tb2_out=getimage(h);
if isempty(tb2_out)
    return;
end

[file,path] = uiputfile('*.jpg','Sauvegarder image filtree');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
imwrite(tb2_out,[path file]);

% --- Executes on selection change in tb2_pop.
function tb2_pop_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_pop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb2_out_t);
cla;
axes(handles.tb2_out_f);
cla;
axes(handles.tb2_filtre_f);
cla;

switch get(findobj('Tag','tb2_pop'),'Value')
    case 1
        set(findobj('Tag','tb2_w1'),'Visible','on');
        set(findobj('Tag','tb2_w2'),'Visible','off');
        set(findobj('Tag','tb2_textgains'),'Visible','off');
        set(findobj('Tag','tb2_gl'),'Visible','off');
        set(findobj('Tag','tb2_gh'),'Visible','off');
        set(findobj('Tag','tb2_pop2'),'Visible','on');
        set(findobj('Tag','tb2_text_w'),'Visible','on');
        set(findobj('Tag', 'tb2_openfilter'), 'Visible', 'off');
    case 2
        set(findobj('Tag','tb2_w1'),'Visible','on');
        set(findobj('Tag','tb2_w2'),'Visible','off');
        set(findobj('Tag','tb2_textgains'),'Visible','off');
        set(findobj('Tag','tb2_gl'),'Visible','off');
        set(findobj('Tag','tb2_gh'),'Visible','off');
        set(findobj('Tag','tb2_pop2'),'Visible','on');
        set(findobj('Tag','tb2_text_w'),'Visible','on');
        set(findobj('Tag', 'tb2_openfilter'), 'Visible', 'off');
    case 3
        set(findobj('Tag','tb2_w1'),'Visible','on');
        set(findobj('Tag','tb2_w2'),'Visible','off');
        set(findobj('Tag','tb2_textgains'),'Visible','on');
        set(findobj('Tag','tb2_gl'),'Visible','on');
        set(findobj('Tag','tb2_gh'),'Visible','on');
        set(findobj('Tag','tb2_pop2'),'Visible','on');
        set(findobj('Tag','tb2_text_w'),'Visible','on');
        set(findobj('Tag', 'tb2_openfilter'), 'Visible', 'off');
    case 4
        set(findobj('Tag','tb2_w1'),'Visible','on');
        set(findobj('Tag','tb2_w2'),'Visible','on');
        set(findobj('Tag','tb2_textgains'),'Visible','off');
        set(findobj('Tag','tb2_gl'),'Visible','off');
        set(findobj('Tag','tb2_gh'),'Visible','off');
        set(findobj('Tag','tb2_pop2'),'Visible','on');
        set(findobj('Tag','tb2_text_w'),'Visible','on');
        set(findobj('Tag', 'tb2_openfilter'), 'Visible', 'off');
    case 5
        set(findobj('Tag','tb2_w1'),'Visible','on');
        set(findobj('Tag','tb2_w2'),'Visible','on');
        set(findobj('Tag','tb2_textgains'),'Visible','off');
        set(findobj('Tag','tb2_gl'),'Visible','off');
        set(findobj('Tag','tb2_gh'),'Visible','off');
        set(findobj('Tag','tb2_pop2'),'Visible','on');
        set(findobj('Tag','tb2_text_w'),'Visible','on');
        set(findobj('Tag', 'tb2_openfilter'), 'Visible', 'off');
    case 6
        % Custom
        set(findobj('Tag','tb2_w1'),'Visible','off');
        set(findobj('Tag','tb2_w2'),'Visible','off');
        set(findobj('Tag','tb2_textgains'),'Visible','off');
        set(findobj('Tag','tb2_gl'),'Visible','off');
        set(findobj('Tag','tb2_gh'),'Visible','off');
        set(findobj('Tag','tb2_pop2'),'Visible','off');
        set(findobj('Tag','tb2_text_w'),'Visible','off');
        set(findobj('Tag', 'tb2_openfilter'), 'Visible', 'on');
    otherwise
end


% --- Executes during object creation, after setting all properties.
function tb2_pop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb2_pop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on selection change in tb2_pop2.
function tb2_pop2_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_pop2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
switch get(findobj('Tag','tb2_pop2'),'Value')
    case 1
        set(findobj('Tag','tb2_n'),'Visible','off');
        set(findobj('Tag','tb2_textordre'),'Visible','off');
    case 2
        set(findobj('Tag','tb2_n'),'Visible','on');
        set(findobj('Tag','tb2_textordre'),'Visible','on');
    case 3
        set(findobj('Tag','tb2_n'),'Visible','off');
        set(findobj('Tag','tb2_textordre'),'Visible','off');
    otherwise
end


% --- Executes during object creation, after setting all properties.
function tb2_pop2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb2_pop2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Butterworth LP
function f = butterworth(sze, cutoff, n)
rows = sze(1);
cols = sze(2);
x =  (ones(rows,1) * [1:cols]  - (fix(cols/2)+1))/cols;
y =  ([1:rows]' * ones(1,cols) - (fix(rows/2)+1))/rows;
radius = sqrt(x.^2 + y.^2);
f = 1 ./ (1.0 + (radius ./ cutoff).^(2*n));



% --- Executes on button press in tb2_filtrer.
function tb2_filtrer_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_filtrer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


h=findobj('Tag','tb2_in_f');
tb2_inf=getimage(h);
maxsp1=max(tb2_inf(:));
minsp1=min(tb2_inf(:));

h=findobj('Tag','tb2_in_t');
tb2_in=getimage(h);
[n m]=size(tb2_in);
PadHeight = 2^nextpow2(n);
PadWidth = 2^nextpow2(m);

switch get(findobj('Tag','tb2_pop'),'Value')
    case 1
        %filtre passe-bas
        w1=str2double(get(findobj('Tag','tb2_w1'),'String'));
        [f1,f2] = freqspace([PadHeight PadWidth],'meshgrid');
        f1=f1/2;
        f2=f2/2;
        switch get(findobj('Tag','tb2_pop2'),'Value')
            case 1
                %ideal
                Hd = zeros(PadHeight,PadWidth); d = sqrt(f1.^2 + f2.^2) < w1;
                Hd(d) = 1;
            case 2
                %butter
                ordre=str2num(get(findobj('Tag','tb2_n'),'String'));
                Hd = butterworth([PadHeight PadWidth], w1, ordre);
            case 3
                %gaussien
                Hd = fspecial('gaussian',[PadHeight PadWidth],w1*min([PadHeight PadWidth]));
                Hd = Hd./max(max(Hd));
            otherwise
        end
        axes(handles.tb2_filtre_f);
        h=imshow(Hd);
        set(h,'Tag','tb2_filtre_f');
        
        If=fftshift(fft2(tb2_in,PadHeight,PadWidth));
        Ifg=If.*Hd;
        Ig=real(ifft2(fftshift(Ifg)));
        Ig_crop=Ig(1:n,1:m);
        axes(handles.tb2_out_t);h=imshow((Ig_crop));
        set(h,'Tag','tb2_out_t');
        axes(handles.tb2_out_f);imshow(log(1+abs(Ifg)),[minsp1 maxsp1]);
    case 2
        % Passe-haut
        w1=str2num(get(findobj('Tag','tb2_w1'),'String'));
        [f1,f2] = freqspace([PadHeight PadWidth],'meshgrid');
        f1=f1/2;
        f2=f2/2;
        switch get(findobj('Tag','tb2_pop2'),'Value')
            case 1
                %ideal
                Hd = zeros(PadHeight,PadWidth); d = sqrt(f1.^2 + f2.^2) < w1;
                Hd(d) = 1;
            case 2
                %butter
                ordre=str2num(get(findobj('Tag','tb2_n'),'String'));
                Hd = butterworth([PadHeight PadWidth], w1, ordre);
            case 3
                %gaussien
                Hd = fspecial('gaussian',[PadHeight PadWidth],w1*min([PadHeight PadWidth]));
                Hd = Hd./max(max(Hd));
            otherwise
        end
        Hd=1-Hd;
        axes(handles.tb2_filtre_f);
        h=imshow(Hd);
        set(h,'Tag','tb2_filtre_f');
        
        If=fftshift(fft2(tb2_in,PadHeight,PadWidth));
        Ifg=If.*Hd;
        Ig=real(ifft2(fftshift(Ifg)));
        Ig_crop=Ig(1:n,1:m);
        axes(handles.tb2_out_t);
        h=imshow((Ig_crop));
        set(h,'Tag','tb2_out_t');
        axes(handles.tb2_out_f);imshow(log(1+abs(Ifg)),[minsp1 maxsp1]);
    case 3
        %homomorphique
        w1=str2num(get(findobj('Tag','tb2_w1'),'String'));
        gl=str2num(get(findobj('Tag','tb2_gl'),'String'));
        gh=str2num(get(findobj('Tag','tb2_gh'),'String'));
        [f1,f2] = freqspace([PadHeight PadWidth],'meshgrid');
        f1=f1/2;
        f2=f2/2;
        switch get(findobj('Tag','tb2_pop2'),'Value')
            case 1
                %ideal
                Hd = zeros(PadHeight,PadWidth); d = sqrt(f1.^2 + f2.^2) < w1;
                Hd(d) = 1;
            case 2
                %butter
                ordre=str2num(get(findobj('Tag','tb2_n'),'String'));
                Hd = butterworth([PadHeight PadWidth], w1, ordre);
            case 3
                %gaussien
                Hd = fspecial('gaussian',[PadHeight PadWidth],w1*min([PadHeight PadWidth]));
                Hd = Hd./max(max(Hd));
            otherwise
        end
        Hd=1-Hd;
        Hd=(gh-gl)*Hd;
        Hd=gl+Hd;
        axes(handles.tb2_filtre_f);
        h=imshow(abs(Hd)/max(max(Hd)));
        set(h,'Tag','tb2_filtre_f');
        
        Iz=find(tb2_in==0);
        tb2_in(Iz)=0.01;
        If=fftshift(fft2(log(double(tb2_in)),PadHeight,PadWidth));
        Ifg=If.*Hd;
        Ig=real(ifft2(fftshift(Ifg)));
        Ig_crop=Ig(1:n,1:m);
        axes(handles.tb2_out_t);
        %h=imshow(exp(Ig_crop)./max(max(exp(Ig_crop))),[]);
        h=imshow(exp(Ig_crop));
        set(h,'Tag','tb2_out_t');
        axes(handles.tb2_out_f);imshow(log(1+abs(Ifg)),[minsp1 maxsp1]);
    case 4
        % Passe bande
        w1_t=str2num(get(findobj('Tag','tb2_w1'),'String'));
        w2_t=str2num(get(findobj('Tag','tb2_w2'),'String'));
        w1=min([w1_t w2_t]);
        w2=max([w1_t w2_t]);
        
        [f1,f2] = freqspace([PadHeight PadWidth],'meshgrid');
        f1=f1/2;
        f2=f2/2;
        switch get(findobj('Tag','tb2_pop2'),'Value')
            case 1
                %ideal
                Hd1 = zeros(PadHeight,PadWidth); d = sqrt(f1.^2 + f2.^2) < w1;
                Hd1(d) = 1;
                Hd2 = zeros(PadHeight,PadWidth); d = sqrt(f1.^2 + f2.^2) < w2;
                Hd2(d) = 1;
            case 2
                %butter
                ordre=str2num(get(findobj('Tag','tb2_n'),'String'));
                Hd1 = butterworth([PadHeight PadWidth], w1, ordre);
                Hd2 = butterworth([PadHeight PadWidth], w2, ordre);
            case 3
                %gaussien
                Hd1 = fspecial('gaussian',[PadHeight PadWidth],w1*min([PadHeight PadWidth]));
                Hd1 = Hd1./max(max(Hd1));
                Hd2 = fspecial('gaussian',[PadHeight PadWidth],w2*min([PadHeight PadWidth]));
                Hd2 = Hd2./max(max(Hd2));
            otherwise
        end
        Hd=1-(1-Hd2+Hd1);
        axes(handles.tb2_filtre_f);
        h=imshow(Hd);
        set(h,'Tag','tb2_filtre_f');
        
        If=fftshift(fft2(tb2_in,PadHeight,PadWidth));
        Ifg=If.*Hd;
        Ig=real(ifft2(fftshift(Ifg)));
        Ig_crop=Ig(1:n,1:m);
        axes(handles.tb2_out_t);
        h=imshow((Ig_crop));
        set(h,'Tag','tb2_out_t');
        axes(handles.tb2_out_f);imshow(log(1+abs(Ifg)),[minsp1 maxsp1]);
    case 5
        % Coupe bande
        w1_t=str2num(get(findobj('Tag','tb2_w1'),'String'));
        w2_t=str2num(get(findobj('Tag','tb2_w2'),'String'));
        w1=min([w1_t w2_t]);
        w2=max([w1_t w2_t]);
        
        [f1,f2] = freqspace([PadHeight PadWidth],'meshgrid');
        f1=f1/2;
        f2=f2/2;
        switch get(findobj('Tag','tb2_pop2'),'Value')
            case 1
                %ideal
                Hd1 = zeros(PadHeight,PadWidth); d = sqrt(f1.^2 + f2.^2) < w1;
                Hd1(d) = 1;
                Hd2 = zeros(PadHeight,PadWidth); d = sqrt(f1.^2 + f2.^2) < w2;
                Hd2(d) = 1;
            case 2
                %butter
                ordre=str2num(get(findobj('Tag','tb2_n'),'String'));
                Hd1 = butterworth([PadHeight PadWidth], w1, ordre);
                Hd2 = butterworth([PadHeight PadWidth], w2, ordre);
            case 3
                %gaussien
                Hd1 = fspecial('gaussian',[PadHeight PadWidth],w1*min([PadHeight PadWidth]));
                Hd1 = Hd1./max(max(Hd1));
                Hd2 = fspecial('gaussian',[PadHeight PadWidth],w2*min([PadHeight PadWidth]));
                Hd2 = Hd2./max(max(Hd2));
            otherwise
        end
        Hd=1-Hd2+Hd1;
        axes(handles.tb2_filtre_f);
        h=imshow(Hd);
        set(h,'Tag','tb2_filtre_f');
        
        If=fftshift(fft2(tb2_in,PadHeight,PadWidth));
        Ifg=If.*Hd;
        Ig=real(ifft2(fftshift(Ifg)));
        Ig_crop=Ig(1:n,1:m);
        axes(handles.tb2_out_t);
        h=imshow((Ig_crop));
        set(h,'Tag','tb2_out_t');
        axes(handles.tb2_out_f);imshow(log(1+abs(Ifg)),[minsp1 maxsp1]);
        
    case 6
        % Custom
        h=findobj('Tag','tb2_filtre_f');
        Hd=getimage(h);
        
        HdSize = size(Hd);
        if HdSize(1) ~= PadHeight || HdSize(2) ~= PadWidth
            Err = MException('SizeChk:WrongSize', ...
                'The filter specified does not have the correct size');
            Cause = MException( 'SizeChk:WrongSize', ...
                'Size should be [%s %s] for this image', ...
                num2str(PadHeight), num2str(PadWidth) );
            Err = addCause(Err, Cause);
            throw(Err);
        end
        
        If=fftshift(fft2(tb2_in,PadHeight,PadWidth));
        Ifg=If.*Hd;
        Ig=real(ifft2(fftshift(Ifg)));
        Ig_crop=Ig(1:n,1:m);
        axes(handles.tb2_out_t);
        h=imshow((Ig_crop));
        set(h,'Tag','tb2_out_t');
        axes(handles.tb2_out_f);imshow(log(1+abs(Ifg)),[minsp1 maxsp1]);
        
    otherwise
end








function tb2_w1_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_w1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb2_w1 as text
%        str2double(get(hObject,'String')) returns contents of tb2_w1 as a double


% --- Executes during object creation, after setting all properties.
function tb2_w1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb2_w1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb2_w2_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_w2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb2_w2 as text
%        str2double(get(hObject,'String')) returns contents of tb2_w2 as a double


% --- Executes during object creation, after setting all properties.
function tb2_w2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb2_w2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb2_n_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb2_n as text
%        str2double(get(hObject,'String')) returns contents of tb2_n as a double


% --- Executes during object creation, after setting all properties.
function tb2_n_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb2_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb2_gl_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_gl (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb2_gl as text
%        str2double(get(hObject,'String')) returns contents of tb2_gl as a double


% --- Executes during object creation, after setting all properties.
function tb2_gl_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb2_gh_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function tb2_gh_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb2_in_t_zoom.
function tb2_in_t_zoom_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_in_t_zoom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb2_in_t');
tb2_in=getimage(h);
figure,imshow(tb2_in)


% --- Executes on button press in tb2_out_t_zoom.
function tb2_out_t_zoom_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_out_t_zoom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb2_out_t');
tb2_out=getimage(h);
figure,imshow(tb2_out)


% --- Executes on button press in tb2_filtre_zoom.
function tb2_filtre_zoom_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_filtre_zoom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb2_filtre_f');
tb2_filtre=getimage(h);
[n m]=size(tb2_filtre);
[f1,f2] = freqspace([n m],'meshgrid');
figure,mesh(f1,f2,tb2_filtre)






% --- Executes on button press in tb3_open.
function tb3_open_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb3_out);
cla;

[FileName,PathName] = uigetfile('*.*','Choisir une image');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end
[in map]=imread([PathName FileName]);
if ~isempty(map)
    in=ind2gray(in, map);
end
if size(in,3)==3
    in=rgb2gray(in);
end
if islogical(in)
    in = cast(in, 'uint8');
    in = in .* 255;
end
in=double(in);
in=in./255;
axes(handles.tb3_in);
h=imshow(in);
set(h,'Tag','tb3_in');

% --- Executes on button press in tb3_in_d_z.
function tb3_in_d_z_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_in_d_z (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb3_in_d');
tb3_in_d=getimage(h);
figure,imshow(tb3_in_d)

% --- Executes on button press in tb3_save.
function tb3_save_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb3_out');
tb3_out=getimage(h);
if isempty(tb3_out)
    return;
end

[file,path] = uiputfile('*.jpg','Sauvegarder image filtree');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
imwrite(tb3_out,[path file]);

% --- Executes on button press in tb3_out_z.
function tb3_out_z_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_out_z (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb3_out');
tb3_out=getimage(h);
figure,imshow(tb3_out)

% --- Executes on selection change in tb3_pop_bruit.
function tb3_pop_bruit_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_pop_bruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb3_pop_bruit contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb3_pop_bruit
switch get(findobj('Tag','tb3_pop_bruit'),'Value')
    case 1
        set(findobj('Tag','tb3_sigmabruit'),'Visible','off');
        set(findobj('Tag','tb3_sigmabruitedit'),'Visible','off');
        set(findobj('Tag','edit4'),'String','Aide');
    case 2
        set(findobj('Tag','tb3_sigmabruit'),'Visible','on');
        set(findobj('Tag','tb3_sigmabruitedit'),'Visible','on');
        set(findobj('Tag','tb3_sigmabruit'),'String','Sigma');
        set(findobj('Tag','edit4'),'String','Bruit gaussien de moyenne 0 et de sigma à choisir');
    case 3
        set(findobj('Tag','tb3_sigmabruit'),'Visible','on');
        set(findobj('Tag','tb3_sigmabruitedit'),'Visible','on');
        set(findobj('Tag','tb3_sigmabruit'),'String','Densité');
        set(findobj('Tag','edit4'),'String','Bruit sel et poivre de densité à choisir');
    otherwise
end

% --- Executes during object creation, after setting all properties.
function tb3_pop_bruit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_pop_bruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in tb3_pop_flou.
function tb3_pop_flou_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_pop_flou (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb3_pop_flou contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb3_pop_flou
%FANTIN
switch get(findobj('Tag','tb3_pop_flou'),'Value')
    case 1
        set(findobj('Tag','tb3_intflou'),'Visible','off');
        set(findobj('Tag','tb3_intflouedit'),'Visible','off');
        set(findobj('Tag','tb3_angleflou'),'Visible','off');
        set(findobj('Tag','tb3_angleflouedit'),'Visible','off');
        set(findobj('Tag','edit4'),'String','Aide');

    case 2
        set(findobj('Tag','tb3_intflou'),'Visible','on');
        set(findobj('Tag','tb3_intflouedit'),'Visible','on');
        set(findobj('Tag','tb3_intflou'),'String','Sigma');
        set(findobj('Tag','tb3_angleflou'),'Visible','off');
        set(findobj('Tag','tb3_angleflouedit'),'Visible','off');
        set(findobj('Tag','edit4'),'String','Flou gaussien de sigma à choisir');

    case 3
        set(findobj('Tag','tb3_intflou'),'Visible','on');
        set(findobj('Tag','tb3_intflouedit'),'Visible','on');
        set(findobj('Tag','tb3_intflou'),'String','Vitesse');
        set(findobj('Tag','tb3_angleflou'),'Visible','on');
        set(findobj('Tag','tb3_angleflouedit'),'Visible','on');
        set(findobj('Tag','edit4'),'String','Flou de déplacement de vitesse et d''angle à choisir');
    otherwise

end


% --- Executes during object creation, after setting all properties.
function tb3_pop_flou_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_pop_flou (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');

end


% --- Executes on button press in tb3_degrader.
function tb3_degrader_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_degrader (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb3_in');
tb3_in=getimage(h);



d_flou= str2double(get(findobj('Tag','tb3_intflouedit'),'String'))+1E-5;
d_angle = str2double(get(findobj('Tag','tb3_angleflouedit'),'String'));
switch get(findobj('Tag','tb3_pop_flou'),'Value')
    case 1
        tb3_in_d=tb3_in;
    case 2
        r = fspecial('gaussian', [max([3 round(d_flou*3)]) max([3 round(d_flou*3)])], d_flou);
        tb3_in_d = imfilter(tb3_in,r,'conv', 'circular' );
    case 3
        r = fspecial('motion', d_flou, d_angle);
        tb3_in_d = imfilter(tb3_in,r,'conv', 'circular' );
    otherwise
        tb3_in_d=tb3_in;
end


d = str2double(get(findobj('Tag','tb3_sigmabruitedit'),'String'));
switch get(findobj('Tag','tb3_pop_bruit'),'Value')
    case 1
        tb3_in_d2=tb3_in_d;
    case 2
        tb3_in_d2=imnoise(tb3_in_d,'gaussian',0,d);
    case 3
        tb3_in_d2=imnoise(tb3_in_d,'salt & pepper',d);
    otherwise
        tb3_in_d2=tb3_in_d;
end


axes(handles.tb3_in_d);
h=imshow(tb3_in_d2);
set(h,'Tag','tb3_in_d');

set(findobj('Tag','tb3_restaurer'),'Visible','on');






% --- Executes on selection change in tb3_pop_restauration.
function tb3_pop_restauration_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_pop_restauration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb3_pop_restauration contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb3_pop_restauration


% --- Executes during object creation, after setting all properties.
function tb3_pop_restauration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_pop_restauration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in tb3_pop_debruiter.
function tb3_pop_debruiter_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_pop_debruiter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns tb3_pop_debruiter contents as cell array
%        contents{get(hObject,'Value')} returns selected item from tb3_pop_debruiter


% --- Executes during object creation, after setting all properties.
function tb3_pop_debruiter_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_pop_debruiter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb3_n_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_n as text
%        str2double(get(hObject,'String')) returns contents of tb3_n as a double


% --- Executes during object creation, after setting all properties.
function tb3_n_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_n (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb3_m_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_m (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_m as text
%        str2double(get(hObject,'String')) returns contents of tb3_m as a double


% --- Executes during object creation, after setting all properties.
function tb3_m_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_m (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb3_sigma_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_sigma as text
%        str2double(get(hObject,'String')) returns contents of tb3_sigma as a double


% --- Executes during object creation, after setting all properties.
function tb3_sigma_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_sigma (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb3_seuil_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_seuil (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_seuil as text
%        str2double(get(hObject,'String')) returns contents of tb3_seuil as a double


% --- Executes during object creation, after setting all properties.
function tb3_seuil_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_seuil (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb3_restaurer.
function tb3_restaurer_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_restaurer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h=findobj('Tag','tb3_in_d');
tb3_in=getimage(h);
[n,m]=size(tb3_in);


d_flou= str2double(get(findobj('Tag','tb3_intflouedit'),'String'))+1E-5;
d_angle = str2double(get(findobj('Tag','tb3_angleflouedit'),'String'));
switch get(findobj('Tag','tb3_pop_flou'),'Value')
    case 1
        r = 1;
    case 2
        r = fspecial('gaussian', [max([3 round(d_flou*3)]) max([3 round(d_flou*3)])], d_flou);
    case 3
        r = fspecial('motion', d_flou, d_angle);
    otherwise
end

switch get(findobj('Tag','tb3_poprestauration'),'Value')
    case 1
        Ig_crop=tb3_in;
    case 2
%         Hd=fftshift(fft2(r,2*n,2*m));
%         If=fftshift(fft2(tb3_in,2*n,2*m));
%         Hdd=ones(2*n,2*m);
%         Ihd=find(Hd>0.01);
%         Hdd(Ihd)=Hd(Ihd);
%         Ifg=If./Hdd;
%         Ig=ifft2(fftshift(Ifg));
%         Ig_crop=Ig(1:n,1:m);
        Ig_crop=deconvwnr(tb3_in,r,0.0);
    case 3
        Kr=str2double(get(findobj('Tag','tb3_seuil'),'String'));
        Ig_crop=deconvwnr(tb3_in,r,Kr);
    otherwise
end
axes(handles.tb3_out);h=imshow(Ig_crop,[]);
set(h,'Tag','tb3_out');





% --- Executes on slider movement.
function tb3_slider_bruit_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_slider_bruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function tb3_slider_bruit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_slider_bruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function tb3_slider_flou_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_slider_flou (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function tb3_slider_flou_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_slider_flou (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function tb1_slider_bruit_Callback(hObject, eventdata, handles)
% hObject    handle to tb1_slider_bruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
h=findobj('Tag','tb2_in_t');
tb1_in=getimage(h(size(h,1)));

d=get(findobj('Tag','tb1_slider_bruit'),'Value');
switch get(findobj('Tag','tb1_popaddnoise'),'Value')
    case 1
        tb1_in_n=imnoise(tb1_in,'gaussian',0,0.2*d);
        set(findobj('Tag','tb1_text_sigma'),'String',['Bruit Gaussien de moyenne 0 et de sigma ' num2str(0.2*d)]);
    case 2
        tb1_in_n=imnoise(tb1_in,'salt & pepper',d);
        set(findobj('Tag','tb1_text_sigma'),'String',['Bruit Poivre et sel de densité ' num2str(d)]);
    otherwise
end

axes(handles.tb1_in);
h=imshow(tb1_in_n);
set(h,'Tag','tb1_in');
axes(handles.tb2_in_t);
h=imshow(tb1_in);
set(h,'Tag','tb2_in_t');

% --- Executes during object creation, after setting all properties.
function tb1_slider_bruit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb1_slider_bruit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on selection change in tb3_poprestauration.
function tb3_poprestauration_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_poprestauration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb3_out);
cla;

switch get(findobj('Tag','tb3_poprestauration'),'Value')
    case 1
    case 2
        set(findobj('Tag','tb3_textseuil'),'Visible','off');
        set(findobj('Tag','tb3_seuil'),'Visible','off');
    case 3
        set(findobj('Tag','tb3_textseuil'),'Visible','on');
        set(findobj('Tag','tb3_seuil'),'Visible','on');
    otherwise
end



% --- Executes during object creation, after setting all properties.
function tb3_poprestauration_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_poprestauration (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb4_open.
function tb4_open_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb4_out);
cla;
[FileName,PathName] = uigetfile('*.*','Choisir une image');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end
[in map]=imread([PathName FileName]);
if ~isempty(map)
    in=ind2gray(in, map);
end
if size(in,3)==3
    in=rgb2gray(in);
end
if islogical(in)
    in = cast(in, 'uint8');
    in = in .* 255;
end
in=double(in);
in=in./255;
axes(handles.tb4_in);
h=imshow(in);
set(h,'Tag','tb4_in');

% --- Executes on button press in tb4_in_z.
function tb4_in_z_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_in_z (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb4_in');
tb4_in=getimage(h);
figure,imshow(tb4_in)

% --- Executes on button press in tb4_save.
function tb4_save_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb4_out');
tb4_out=getimage(h(size(h,1)));
if isempty(tb4_out)
    return;
end

[file,path] = uiputfile('*.jpg','Sauvegarder image filtree');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
imwrite(tb4_out,[path file]);


% --- Executes on button press in tb4_out_z.
function tb4_out_z_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_out_z (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb4_out');
tb4_out=getimage(h(size(h,1)));
figure,imshow(tb4_out)

% --- Executes on selection change in tb4_popmethodes.
function tb4_popmethodes_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_popmethodes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb4_out);
cla;


h=findobj('Tag','tb4_in');
tb4_in=getimage(h);
[n m]=size(tb4_in);

th=get(findobj('Tag','tb4_slider'),'Value');

set(findobj('Tag','edit3'),'Visible','off');

switch get(findobj('Tag','tb4_popmethodes'),'Value')
    case 1
        set(findobj('Tag','tb4_segmenter'),'Visible','off');
        set(findobj('Tag','tb4_slider'),'Visible','off');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','off');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','off');
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Segmentation: Choisir une méthode');
    case 2
        
        tb4_out=edge(tb4_in,'sobel',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
        
        set(findobj('Tag','tb4_slider'),'Visible','on');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','on');
        set(findobj('Tag','tb4_segmenter'),'Visible','off');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','on');
        
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Filtre de Sobel : Charger une image ; Choisir le seuil');
    case 3
        tb4_out=edge(tb4_in,'prewitt',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
        
        set(findobj('Tag','tb4_slider'),'Visible','on');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','on');        
        set(findobj('Tag','tb4_segmenter'),'Visible','off');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','on');
        
        set(findobj('Tag','tb4_Bouton_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Bouton_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Filtre de Prewitt : Charger une image ; Choisir le seuil');
    case 4
        tb4_out=edge(tb4_in,'log',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
        
        set(findobj('Tag','tb4_slider'),'Visible','on');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','on');
        set(findobj('Tag','tb4_segmenter'),'Visible','off');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','on');
        
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Filtre de Marr-Hildreth (Log) : Charger une image ; Choisir le seuil');
    case 5  %canny
        tb4_out=edge(tb4_in,'canny',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
        
        
        set(findobj('Tag','tb4_slider'),'Visible','on');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','on');
        set(findobj('Tag','tb4_segmenter'),'Visible','off');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','on');
        
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Filtre de Canny : Charger une image ; Choisir le seuil');
    case 6
        set(findobj('Tag','tb4_slider'),'Visible','on');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','on');
        set(findobj('Tag','tb4_segmenter'),'Visible','on');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','on');
        set(findobj('Tag','tb4_textseuilslider'),'String','Seuil Canny');
        %         Hough : choix de la methode et des rayons
        set(findobj('Tag','tb4_Cercles'),'Visible','on');
        set(findobj('Tag','tb4_Cercles'),'Value',0);
        set(findobj('Tag','tb4_Droites'),'Value',1);
        set(findobj('Tag','tb4_Droites'),'Visible','on');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','on');
        set(findobj('Tag','tb4_npeak'),'Visible','on');
        set(findobj('Tag','tb4_textnpeak'),'Visible','on');
        set(findobj('Tag','tb4_npeak'),'String','50');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','1) Choisir droite ou cercle; 2) fixer le seuil du canny; 3) Fixer le nombre de pics');
    case 7
        set(findobj('Tag','tb4_slider'),'Visible','off');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','off');
        set(findobj('Tag','tb4_segmenter'),'Visible','on');
        set(findobj('Tag','tb4_seuil'),'Visible','on');
        set(findobj('Tag','tb4_textseuil'),'Visible','on');
        set(findobj('Tag','tb4_histo'),'Visible','on');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','off');
        
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Seuillage semi-automatique : Charger une image ; Choisir un seuil de départ (cliquer sur histo pour voir l''histogramme)');
    case 8
        set(findobj('Tag','tb4_slider'),'Visible','off');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','off');
        set(findobj('Tag','tb4_segmenter'),'Visible','on');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','on');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','off');
        
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Seuillage automatique d''Otsu : Charger une image puis OK');
    case 9
        set(findobj('Tag','tb4_slider'),'Visible','off');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','off');
        set(findobj('Tag','tb4_segmenter'),'Visible','on');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','off');
        
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Split and Merge : Charger une image puis OK');
    case 10
        set(findobj('Tag','tb4_slider'),'Visible','off');
        set(findobj('Tag','tb4_Valeur_Slider'),'Visible','off');
        set(findobj('Tag','tb4_segmenter'),'Visible','on');
        set(findobj('Tag','tb4_seuil'),'Visible','off');
        set(findobj('Tag','tb4_textseuil'),'Visible','off');
        set(findobj('Tag','tb4_histo'),'Visible','off');
        set(findobj('Tag','tb4_textseuil2'),'Visible','off');
        set(findobj('Tag','tb4_textseuilslider'),'Visible','off');
        
        set(findobj('Tag','tb4_Cercles'),'Visible','off');
        set(findobj('Tag','tb4_Droites'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_Rmin'),'Visible','off');
        set(findobj('Tag','tb4_Rmax'),'Visible','off');
        set(findobj('Tag','tb4_panel_bouton'),'Visible','off');
        set(findobj('Tag','tb4_npeak'),'Visible','off');
        set(findobj('Tag','tb4_textnpeak'),'Visible','off');
        set(findobj('Tag','edit3'),'Visible','on');
        set(findobj('Tag','edit3'),'String','Watershed : Charger une image puis OK');
    otherwise
end


% --- Executes during object creation, after setting all properties.
function tb4_popmethodes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_popmethodes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb4_segmenter.
function tb4_segmenter_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_segmenter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb4_out);
cla;

h=findobj('Tag','tb4_in');
tb4_in=getimage(h);
[n m]=size(tb4_in);

switch get(findobj('Tag','tb4_popmethodes'),'Value')
    case 1
    case 2
    case 3
    case 4
    case 5
    case 6
        % Hough
%         Path = cd;
%         addpath([Path '\HouchCircle']);
        thcanny=get(findobj('Tag','tb4_slider'),'Value');
        set(findobj('Tag','tb4_Valeur_Slider'),'String',num2str(1*thcanny));
        BW = edge(tb4_in,'canny',1*thcanny);
        
        NPeak = str2double(get(findobj('Tag','tb4_npeak'),'String'));
        
        if (get(findobj('Tag','tb4_Cercles'),'Value') == get(findobj('Tag','tb4_Cercles'),'Max'))
            
            %lama=0
            % Specifier le rayon des cercles
            radii = str2double(get(findobj('Tag','tb4_Rmin'),'String')):str2double(get(findobj('Tag','tb4_Rmax'),'String'));
            h = circle_hough(BW, radii, 'same', 'normalise');
            
            
            peaks = circle_houghpeaks(h, radii, 'nhoodxy', 15, 'nhoodr', 21, 'npeaks', NPeak);
            axes(handles.tb4_out);
   
            tb4_in_RGB(:,:,1) = double(BW);
            tb4_in_RGB(:,:,2) = double(BW);
            tb4_in_RGB(:,:,3) = double(BW);
            
%             h2 = imshow(tb4_in);
%             hold on;
            for peak = peaks
                [x, y] = circlepoints(peak(3));
                YIndices = y + peak(2);
                YIndices = min( max( YIndices, 1 ), n );
                XIndices = x + peak(1);
                XIndices = min( max( XIndices, 1 ), m );
                for circlecoord=1:size(YIndices,2)
                    tb4_in_RGB( YIndices(1,circlecoord),XIndices(1,circlecoord),1) = 1 ;
                    tb4_in_RGB( YIndices(1,circlecoord),XIndices(1,circlecoord),2) = 0 ;
                    tb4_in_RGB( YIndices(1,circlecoord),XIndices(1,circlecoord),3) = 0;
                end
%                    plot(x+peak(1), y+peak(2), 'g-');
            end
            h2 = imshow(tb4_in_RGB);
%             hold off
            set(h2,'Tag','tb4_out');
        else
            %lama=1
            [H,T,R] = hough(BW,'RhoResolution',1,'ThetaResolution',1);
            figure,imshow(imadjust(mat2gray(H)),'XData',T,'YData',R,...
                'InitialMagnification','fit');
            title('Transformee de Hough');
            xlabel('\theta'), ylabel('\rho');
            axis on, axis normal, hold on;
            colormap(hot);
            
            P  = houghpeaks(H,NPeak,'threshold',ceil(0.1*max(H(:))), 'NHoodSize', [5 5]);
            x = T(P(:,2)); y = R(P(:,1));
            plot(x,y,'o','color','blue','MarkerSize',12,'LineWidth',4);
            % Find lines and plot them
            lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',7);
            
            fig = figure( 'Visible', 'off' );
            imshow(BW, 'Border', 'tight'), hold on,
            max_len = 0;
            for k = 1:length(lines)
                xy = [lines(k).point1; lines(k).point2];
                plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
                
                % Plot beginnings and ends of lines
                plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
                plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
                
                % Determine the endpoints of the longest line segment
                len = norm(lines(k).point1 - lines(k).point2);
                if ( len > max_len)
                    max_len = len;
                    xy_long = xy;
                end
            end
            
            % highlight the longest line segment
            plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');

            [tb4_out] = export_fig('-native', '-a1', gca );

            close(fig);
            
            axes(handles.tb4_out);
            h = imshow( tb4_out );
            set(h,'Tag','tb4_out');
            
            % Don't forget the hold off if you do hold on!!!!
            hold off;
        end
        
        
    case 7
        % seuil semi auto
        th1=0;
        th2=str2num(get(findobj('Tag','tb4_seuil'),'String'));
        
        while abs(th1-th2)>0.1
            th1=th2;
            ma=mean2(tb4_in(find(tb4_in<=th1)));
            mb=mean2(tb4_in(find(tb4_in>th1)));
            th2=0.5*(ma+mb);
        end
        tb4_out = im2bw(tb4_in,th2);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
        set(findobj('Tag','tb4_textseuil2'),'String',['seuil final = ' num2str(th2)]);
        set(findobj('Tag','tb4_textseuil2'),'Visible','on');
        
    case 8
        % seuil otsu
        th=graythresh(tb4_in);
        tb4_out = im2bw(tb4_in,th);
        
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
        set(findobj('Tag','tb4_textseuil2'),'String',['seuil final = ' num2str(th)]);
        set(findobj('Tag','tb4_textseuil2'),'Visible','on');
    case 9
%         tb4_out=splitmerge(tb4_in,2,@predicate);
%         axes(handles.tb4_out);
%         h=imshow((tb4_out));
        tb4_temp = tb4_in;
        tb4_in = imresize(tb4_temp, [2^(ceil(log2(size(tb4_temp,1)))) 2^(ceil(log2(size(tb4_temp,2))))]);
        
        S = qtdecomp(tb4_in,.2);
        blocks = repmat(uint8(0),size(S));

        for dim = [512 256 128 64 32 16 8 4 2];    
            numblocks = length(find(S==dim));    
            if (numblocks > 0)
                values = repmat(uint8(1),[dim dim numblocks]);
                values(2:dim,2:dim,:) = 0;
                blocks = qtsetblk(blocks,S,dim,values);
            end
        end
        
        blocks(end,1:end) = 1;
        blocks(1:end,end) = 1;
%          whos tb4_in;
%          max(max(tb4_in))
        tb4_out = imresize(blocks*255.0+uint8(255.0*tb4_in), [size(tb4_temp,1) size(tb4_temp,2)]);
       
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        
        set(h,'Tag','tb4_out');
    case 10
        hy = fspecial('sobel');
        hx = hy';
        Iy = imfilter(double(tb4_in), hy, 'replicate');
        Ix = imfilter(double(tb4_in), hx, 'replicate');
        gradmag = sqrt(Ix.^2 + Iy.^2);
        
        se = strel('disk', 20);
        Io = imopen(tb4_in, se);
        Ie = imerode(tb4_in, se);
        Iobr = imreconstruct(Ie,tb4_in);
        Ioc = imclose(Io, se);
        Iobrd = imdilate(Iobr, se);
        Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
        Iobrcbr = imcomplement(Iobrcbr);
        fgm = imregionalmax(Iobrcbr);
        I2 = tb4_in;
        I2(fgm) = 255;
        se2 = strel(ones(5,5));
        fgm2 = imclose(fgm, se2);
        fgm3 = imerode(fgm2, se2);
        fgm4 = bwareaopen(fgm3, 20);
        I3 = tb4_in;
        I3(fgm4) = 255;
        
        bw = im2bw(Iobrcbr, graythresh(Iobrcbr));
        D = bwdist(bw);
        DL = watershed(D);
        bgm = DL == 0;
        gradmag2 = imimposemin(gradmag, bgm | fgm4);
        L = watershed(gradmag2);
        I4 = tb4_in;
        I4(imdilate(L == 0, ones(3, 3)) | bgm | fgm4) = 255;
        Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
        
        % Mix the two images together. This will ensure
        % that the watershed is displayed when the zoom or save function
        % are called.
        Lrgb = im2double( Lrgb );
        Temp = repmat( tb4_in, [1 1 3] );
        Temp = 0.7 * Temp + 0.3 * Lrgb;
        
        axes(handles.tb4_out);
        h=imshow(Temp); hold on
        set(h,'Tag','tb4_out');
        
        % Don't forget the hold off if you do hold on!!!!
        hold off;
    otherwise
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function flag=predicate(region)

sd=std2(region);
m=mean2(region);
flag= (sd>0.05) &(m>0.1);


% --- Executes on slider movement.
function tb4_slider_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb4_out);
cla;

h=findobj('Tag','tb4_in');
tb4_in=getimage(h);
[n m]=size(tb4_in);

th=get(findobj('Tag','tb4_slider'),'Value');

switch get(findobj('Tag','tb4_popmethodes'),'Value')
    case 1
    case 2
        set(findobj('Tag','tb4_Valeur_Slider'),'String',num2str(1*th));
        tb4_out=edge(tb4_in,'sobel',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
    case 3
        set(findobj('Tag','tb4_Valeur_Slider'),'String',num2str(1*th));
        tb4_out=edge(tb4_in,'prewitt',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
    case 4
        set(findobj('Tag','tb4_Valeur_Slider'),'String',num2str(1*th));
        tb4_out=edge(tb4_in,'log',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
    case 5
        set(findobj('Tag','tb4_Valeur_Slider'),'String',num2str(1*th));
        tb4_out=edge(tb4_in,'canny',1*th);
        axes(handles.tb4_out);
        h=imshow((tb4_out));
        set(h,'Tag','tb4_out');
    case 6
        set(findobj('Tag','tb4_Valeur_Slider'),'String',num2str(1*th));
        tb4_out=edge(tb4_in,'canny',1*th);
        h2 = imshow(tb4_out);
        set(h2,'Tag','tb4_out');
    case 7
    case 8
    case 9
    case 10
    otherwise
end


% --- Executes during object creation, after setting all properties.
function tb4_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in tb4_histo.
function tb4_histo_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_histo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb4_in');
tb4_in=getimage(h);
figure,imhist(tb4_in);
title('histogramme image originale')


function tb4_seuil_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_seuil (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb4_seuil as text
%        str2double(get(hObject,'String')) returns contents of tb4_seuil as a double


% --- Executes during object creation, after setting all properties.
function tb4_seuil_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_seuil (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb4_Droites.
function tb4_Droites_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_Droites (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tb4_Droites
if get(hObject,'Value')==1
%lama= 5
set(findobj('Tag','tb4_Text_Rmin'),'Visible','off');
set(findobj('Tag','tb4_Text_Rmax'),'Visible','off');
set(findobj('Tag','tb4_Rmin'),'Visible','off');
set(findobj('Tag','tb4_Rmax'),'Visible','off');
set(findobj('Tag','tb4_Cercles'),'Value',0)
end


% --- Executes on button press in tb4_Cercles.
function tb4_Cercles_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_Cercles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of tb4_Cercles
if get(hObject,'Value')==1
%lama= 6
set(findobj('Tag','tb4_Text_Rmin'),'Visible','on');
set(findobj('Tag','tb4_Text_Rmax'),'Visible','on');
set(findobj('Tag','tb4_Rmin'),'Visible','on');
set(findobj('Tag','tb4_Rmax'),'Visible','on');
set(findobj('Tag','tb4_Droites'),'Value',0)
end 



function tb4_Valeur_Slider_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_Valeur_Slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb4_Valeur_Slider as text
%        str2double(get(hObject,'String')) returns contents of tb4_Valeur_Slider as a double


% --- Executes during object creation, after setting all properties.
function tb4_Valeur_Slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_Valeur_Slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb4_npeak_Callback(hObject, eventdata, handles)
% hObject    handle to tb4_npeak (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb4_npeak as text
%        str2double(get(hObject,'String')) returns contents of tb4_npeak as a double


% --- Executes during object creation, after setting all properties.
function tb4_npeak_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb4_npeak (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in tb2_openfilter.
function tb2_openfilter_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_openfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.tb2_filtre_f);
cla;
[FileName,PathName] = uigetfile('*.*','Choisir une image');
if (~ischar(FileName) && FileName == 0) || (~ischar(PathName) && PathName == 0)
    return;
end
[in map]=imread([PathName FileName]);
if ~isempty(map)
    in=ind2gray(in, map);
end
if size(in,3)==3
    in=rgb2gray(in);
end
if islogical(in)
    in = cast(in, 'uint8');
    in = in .* 255;
end
in=double(in);
in=in./255;
h = imshow(in);
set(h,'Tag','tb2_filtre_f');


% --- Executes on button press in tb2_in_f_save.
function tb2_in_f_save_Callback(hObject, eventdata, handles)
% hObject    handle to tb2_in_f_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=findobj('Tag','tb2_in_f');
tb2_in_f=getimage(h);
if isempty(tb2_in_f)
    return;
end

[file,path] = uiputfile('*.png','Sauvegarder transformée de Fourier');
if (~ischar(file) && file == 0) || (~ischar(path) && path == 0)
    return;
end
imwrite(tb2_in_f,[path file]);



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb3_intflouedit_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_intflouedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_intflouedit as text
%        str2double(get(hObject,'String')) returns contents of tb3_intflouedit as a double


% --- Executes during object creation, after setting all properties.
function tb3_intflouedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_intflouedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb3_angleflouedit_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_angleflouedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_angleflouedit as text
%        str2double(get(hObject,'String')) returns contents of tb3_angleflouedit as a double


% --- Executes during object creation, after setting all properties.
function tb3_angleflouedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_angleflouedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tb3_sigmabruitedit_Callback(hObject, eventdata, handles)
% hObject    handle to tb3_sigmabruitedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tb3_sigmabruitedit as text
%        str2double(get(hObject,'String')) returns contents of tb3_sigmabruitedit as a double


% --- Executes during object creation, after setting all properties.
function tb3_sigmabruitedit_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tb3_sigmabruitedit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
