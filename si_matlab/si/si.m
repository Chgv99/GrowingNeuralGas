dataDir = fullfile(pwd,'data');
vinosFile = fullfile(dataDir,'Clasificacion_Vinos.csv');
cientesFile = fullfile(dataDir,'Clientes_Ventas_por_Mayor.csv');
paisesFile = fullfile(dataDir,'Datos_Paises.csv');
sampleFile = fullfile(dataDir,'Sample_Cluster_Data_2D.csv');
disp('Vinos')
vinos = readtable(vinosFile);
disp('Clientes')
clientes = readtable(cientesFile);
disp('Paises')
paises = readtable(paisesFile);
% paises = readtable(paisesFile,'VariableNamingRule','preserve');
disp('Sample')
sampleFile =  fullfile(sampleFile);
sample = readtable(sampleFile);

name = "vinos_todas-PCA_3-escalado-a_0.5-amax_8-n_0.05-e_10-31-12-2021-17-30-02";
matDir = fullfile(pwd,name);

aFile = fullfile(matDir,'matlab_a.csv');
allFileProfe = fullfile(matDir,'matlab_all_profe.json');
allFileAlumno = fullfile(matDir,'matlab_all_alumno.json');
idsFile = fullfile(matDir,'matlab_ids.json');
vecinosProfeFile = fullfile(matDir,'matlab_grupos_profe.json'); % grupos profe, vecinos
vecinosAlumnoFile = fullfile(matDir,'matlab_grupos_alumno.json');
nFile = fullfile(matDir,'n.json');

A = readmatrix(aFile);
allProfe = jsondecode(fileread(allFileProfe));
allAlumno= jsondecode(fileread(allFileAlumno));
ids = jsondecode(fileread(idsFile));
vecinosProfe = jsondecode(fileread(vecinosProfeFile));
vecinosAlumno = jsondecode(fileread(vecinosAlumnoFile));
n = jsondecode(fileread(nFile));

allProfe = toCells(allProfe);
allAlumno = toCells(allAlumno);

data = clientes;

[coeff,score,latent,tsquared,explained,mu] = pca(table2array(data));
[coeff,score,latent,tsquared,explained,mu] = pca(normalize(table2array(vinos),'scale'));
% writematrix(score,fullfile(pwd,'dataPCA.csv')) 
data = readmatrix('vinos_todos-PCA_3-escalado.csv');

dataPlot = data;
APlot = A;
dims = 3;
if 2 == size(APlot,2)
    dims = 2;
end
groups = allAlumno;
colors = flipud(hsv(length(groups)));
legendNames = [];

if dims == 2
    aux = scatter(dataPlot(:,1),dataPlot(:,2),...
        'ko', 'filled',...
        'DisplayName', 'Dataset');
    legendNames = zeros(1,length(groups) + 1);
    legendNames(1) = aux;
    hold on
    for i = 1:length(groups)
        % +1 por el indexado de matlab
%         quizás se podría sumar antes para claridad?
        this = cellfun(@(x) x{1} +1 ,groups{i});
        pos = APlot(this,:);
        aux = plot(pos(:,1),pos(:,2),...
        '.',...
        'Color', colors(i,:),...
        'LineStyle', 'none',...
        'DisplayName', 'Grupo'+string(i));
        legendNames(i+1) = aux;
        for j = 1:length(groups{i})
            % +1 por el indexado de matlab
            origin = groups{i}{j}{1}+1;
            for k = 1:length(groups{i}{j}{2})
                % +1 por el indexado de matlab
                dest = groups{i}{j}{2}(k)+1;
                if true
                    line([APlot(origin,1),APlot(dest,1)],...
                        [APlot(origin,2), APlot(dest,2)],...
                        'Color', colors(i,:))
                else  
                    plot([APlot(origin,1), APlot(dest,1)],...
                        [APlot(origin,2), APlot(dest,2)],...
                        '.-',...
                        'Color', colors(i,:))
                end
            end
        end
    end
    
elseif dims == 3
    aux = scatter3(dataPlot(:,1),dataPlot(:,2),dataPlot(:,3),...
        'ko', 'filled',...
        'DisplayName', 'Dataset');
    legendNames = zeros(1,length(groups) + 1);
    legendNames(1) = aux;
    hold on
    for i = 1:length(groups)
        % +1 por el indexado de matlab
%         quizás se podría sumar antes para claridad?
        this = cellfun(@(x) x{1} +1 ,groups{i});
        pos = APlot(this,:);
        aux = plot3(pos(:,1), pos(:,2), pos(:,3),...
        '.',...
        'Color', colors(i,:),...
        'LineStyle', 'none',...
        'DisplayName', 'Grupo'+string(i));
        legendNames(i+1) = aux;
        for j = 1:length(groups{i})
            % +1 por el indexado de matlab
            origin = groups{i}{j}{1}+1;
            for k = 1:length(groups{i}{j}{2})
                % +1 por el indexado de matlab
                dest = groups{i}{j}{2}(k)+1;
                if true
                    line([APlot(origin,1),APlot(dest,1)],...
                        [APlot(origin,2), APlot(dest,2)],...
                        [APlot(origin,3), APlot(dest,3)],...
                        'Color', colors(i,:))
                else  
                    plot3([APlot(origin,1), APlot(dest,1)],...
                        [APlot(origin,2), APlot(dest,2)],...
                        [APlot(origin,3), APlot(dest,3)],...
                        '.-',...
                        'Color', colors(i,:))
                end
            end
        end
    end
    zlabel('PCA3')
end
xlabel('PCA1')
ylabel('PCA2')
legend(legendNames,'location','best')
title(name,'Interpreter','none')
hold off


function jsonlist = toCells(jsonlist)
    for i = 1:length(jsonlist)
        if class(jsonlist{i}) ~= "cell"
            jsonlist{i} = {{jsonlist{i}(1);jsonlist{i}(2)}};
        end
        for j = 1:length(jsonlist{i})
            if class(jsonlist{i}{j}) ~= "cell"
                jsonlist{i}{j} = {jsonlist{i}{j}(1);jsonlist{i}{j}(2)};
            end
        end
    end
end

