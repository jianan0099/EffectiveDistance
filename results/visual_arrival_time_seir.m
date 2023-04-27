effective_dis = readtable('seir_final_results.xlsx','Sheet', 'effective_dis','PreserveVariableNames',true);
infected_time = readtable('seir_final_results.xlsx','Sheet', 'infected_time','PreserveVariableNames',true);
node_type = table2cell(readtable('seir_final_results.xlsx','Sheet', 'node_type','PreserveVariableNames',false));
iso_code = readtable('seir_final_results.xlsx','Sheet', 'iso_code','PreserveVariableNames',true);
p_all = {'0', '0.1', '0.5', '0.95'};
titles = {'p=0', 'p=0.1', 'p=0.5', 'p=0.95'};
figure('Position', [345,170,516,558])
font_size = 12;
texts = char(98:120);
node_type_china = [];
node_type_1 = [];
node_type_5 = [];
node_type_95 = [];
node_type_100 = [];
china_color = [227/255 26/255 28/255];
color_1 = [33/255 113/255 181/255];
color_5 = [158/255 202/255 225/255];
color_95 = [223/255 235/255 247/255];
color_100 = [247/255 251/255 255/255];

for i=2:length(node_type)
    if strcmp(node_type(i),'china')
        node_type_china = [node_type_china,int8(i-1)];
    end
    
    if strcmp(node_type(i),'0.1')
        node_type_1 = [node_type_1,int8(i-1)];
    end
    
    if strcmp(node_type(i),'0.5')
        node_type_5 = [node_type_5,int8(i-1)];
    end
    
    if strcmp(node_type(i),'0.95')
        node_type_95 = [node_type_95,int8(i-1)];
    end
    
    if strcmp(node_type(i),'1')
        node_type_100 = [node_type_100,int8(i-1)];
    end
    
end

for row=1:2
    for col=1:2
        fig_num = (row-1)*2+col;
        subplot(2,2,fig_num,'Position', [0.1+(col-1)*0.48, 0.58-(row-1)*0.48, 0.38, 0.35],'Units','normalized')

        col_name_overall = string(p_all(fig_num));
        infected_time_p = infected_time.(col_name_overall);
        effective_dis_p = effective_dis.(col_name_overall);
        
        
        scatter(infected_time_p(node_type_china),effective_dis_p(node_type_china), 180,'MarkerEdgeColor','black','MarkerFaceColor',china_color)
        hold on
        
        scatter(infected_time_p(node_type_1),effective_dis_p(node_type_1), 130,'MarkerEdgeColor','black','MarkerFaceColor',color_1)
        hold on
        
        scatter(infected_time_p(node_type_5),effective_dis_p(node_type_5), 60,'MarkerEdgeColor','black','MarkerFaceColor',color_5)
        hold on
        
        scatter(infected_time_p(node_type_95),effective_dis_p(node_type_95), 20,'MarkerEdgeColor','black','MarkerFaceColor',color_95)
        hold on
        
        scatter(infected_time_p(node_type_100),effective_dis_p(node_type_100), 20,'MarkerEdgeColor','black','MarkerFaceColor',color_100)
        hold on
        
        set(gca,'FontSize',font_size)
        
        %xlim([0 8.5])
        %xticks([0 2 4 6 8])
        xlim([-0.5 30])
        xticks([0 10 20 30])
        title(titles(fig_num))
        
        if col==1
        ylabel('D')
        end
        
        if row==2
        xlabel('T^a [days]')
        end
        text(-0.15, 1.15, texts(fig_num), 'Units', 'Normalized','FontSize',font_size,'FontWeight','bold');
        
        ylim([-0.5 60])
        grid on
    end
end
saveas(gcf,'seir_eff_fig','epsc')