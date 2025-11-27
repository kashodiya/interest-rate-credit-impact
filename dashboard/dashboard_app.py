"""
DashboardApp class for Plotly Dash web interface.
"""

from dash import Dash, html, dcc
import plotly.graph_objects as go
from typing import Any


class DashboardApp:
    """Main Plotly Dash application coordinating all visualizations."""
    
    def __init__(self, analysis_results: Any):
        """
        Initialize dashboard with analysis results.
        
        Args:
            analysis_results: Object containing all analysis results including
                            merged data, EDA results, correlations, regression, forecasts
        """
        self.analysis_results = analysis_results
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
    
    def _setup_layout(self):
        """Set up the main dashboard layout with professional styling."""
        self.app.layout = html.Div([
            # Header Section
            html.Div([
                html.Div([
                    html.H1("📊 Interest Rate & Consumer Credit Analysis",
                           style={
                               'color': '#ffffff',
                               'margin': '0',
                               'fontSize': '2.5rem',
                               'fontWeight': '600',
                               'letterSpacing': '-0.5px'
                           }),
                    html.P("Federal Reserve Economic Data Analysis Dashboard",
                          style={
                              'color': '#e8f4f8',
                              'margin': '10px 0 0 0',
                              'fontSize': '1.1rem',
                              'fontWeight': '300'
                          })
                ], style={
                    'maxWidth': '1400px',
                    'margin': '0 auto',
                    'padding': '0 20px'
                })
            ], style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'padding': '40px 20px',
                'marginBottom': '30px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
            }),
            
            # Main Content Container
            html.Div([
                # Placeholder containers for panels
                html.Div(id='time-series-container'),
                html.Div(id='correlation-container'),
                html.Div(id='regression-container'),
                html.Div(id='forecast-container'),
                html.Div(id='scenario-container'),
                
                # Footer
                html.Div([
                    html.P([
                        "Data Source: ",
                        html.A("Federal Reserve Economic Data (FRED)", 
                              href="https://fred.stlouisfed.org/",
                              target="_blank",
                              style={'color': '#667eea', 'textDecoration': 'none', 'fontWeight': '500'}),
                        " | Built with Python, Plotly & Dash"
                    ], style={
                        'textAlign': 'center',
                        'color': '#7f8c8d',
                        'fontSize': '0.9rem',
                        'padding': '30px 0',
                        'borderTop': '1px solid #ecf0f1'
                    })
                ])
            ], style={
                'maxWidth': '1400px',
                'margin': '0 auto',
                'padding': '0 20px'
            })
        ], style={
            'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            'backgroundColor': '#f8f9fa',
            'minHeight': '100vh',
            'margin': '0',
            'padding': '0'
        })
    
    def create_time_series_panel(self) -> html.Div:
        """
        Create interactive time series charts panel.
        
        Returns:
            html.Div containing time series visualizations
        """
        # Check if we have merged data
        if not hasattr(self.analysis_results, 'merged_data') or self.analysis_results.merged_data is None:
            return html.Div([
                html.H2("Time Series Data"),
                html.P("No data available. Please run the analysis pipeline first.",
                      style={'color': 'red'})
            ])
        
        df = self.analysis_results.merged_data
        
        # Create interest rates time series
        interest_fig = go.Figure()
        
        # Add traces for interest rates
        if 'fed_funds_rate' in df.columns:
            interest_fig.add_trace(go.Scatter(
                x=df.index, y=df['fed_funds_rate'],
                mode='lines', name='Fed Funds Rate',
                line=dict(color='#e74c3c', width=2)
            ))
        
        if 'treasury_1y' in df.columns:
            interest_fig.add_trace(go.Scatter(
                x=df.index, y=df['treasury_1y'],
                mode='lines', name='Treasury 1Y',
                line=dict(color='#3498db', width=2)
            ))
        
        if 'treasury_10y' in df.columns:
            interest_fig.add_trace(go.Scatter(
                x=df.index, y=df['treasury_10y'],
                mode='lines', name='Treasury 10Y',
                line=dict(color='#2ecc71', width=2)
            ))
        
        if 'prime_rate' in df.columns:
            interest_fig.add_trace(go.Scatter(
                x=df.index, y=df['prime_rate'],
                mode='lines', name='Prime Rate',
                line=dict(color='#f39c12', width=2)
            ))
        
        interest_fig.update_layout(
            title={
                'text': 'Interest Rates Over Time',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Date',
            yaxis_title='Rate (%)',
            hovermode='x unified',
            template='plotly_white',
            height=450,
            margin=dict(l=60, r=40, t=60, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#e9ecef',
                borderwidth=1
            ),
            plot_bgcolor='#fafbfc',
            paper_bgcolor='#ffffff'
        )
        
        # Create consumer credit time series
        credit_fig = go.Figure()
        
        if 'total_credit' in df.columns:
            credit_fig.add_trace(go.Scatter(
                x=df.index, y=df['total_credit'],
                mode='lines', name='Total Credit',
                line=dict(color='#9b59b6', width=2)
            ))
        
        if 'revolving_credit' in df.columns:
            credit_fig.add_trace(go.Scatter(
                x=df.index, y=df['revolving_credit'],
                mode='lines', name='Revolving Credit',
                line=dict(color='#1abc9c', width=2)
            ))
        
        if 'non_revolving_credit' in df.columns:
            credit_fig.add_trace(go.Scatter(
                x=df.index, y=df['non_revolving_credit'],
                mode='lines', name='Non-Revolving Credit',
                line=dict(color='#e67e22', width=2)
            ))
        
        credit_fig.update_layout(
            title={
                'text': 'Consumer Credit Over Time',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Date',
            yaxis_title='Credit (Billions $)',
            hovermode='x unified',
            template='plotly_white',
            height=450,
            margin=dict(l=60, r=40, t=60, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#e9ecef',
                borderwidth=1
            ),
            plot_bgcolor='#fafbfc',
            paper_bgcolor='#ffffff'
        )
        
        # Return panel with both charts in professional card layout
        return html.Div([
            # Section Header
            html.Div([
                html.H2("📈 Time Series Analysis", 
                       style={
                           'color': '#2c3e50',
                           'marginBottom': '10px',
                           'fontSize': '1.8rem',
                           'fontWeight': '600'
                       }),
                html.P("Interactive charts showing interest rates and consumer credit trends over time.",
                      style={'color': '#7f8c8d', 'fontSize': '1rem', 'marginBottom': '25px'})
            ]),
            
            # Interest Rates Card
            html.Div([
                dcc.Graph(
                    id='interest-rates-chart',
                    figure=interest_fig,
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '25px',
                'marginBottom': '25px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            }),
            
            # Consumer Credit Card
            html.Div([
                dcc.Graph(
                    id='consumer-credit-chart',
                    figure=credit_fig,
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '25px',
                'marginBottom': '40px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            })
        ])
    
    def create_correlation_panel(self) -> html.Div:
        """
        Create correlation heatmaps and scatter plots panel.
        
        Returns:
            html.Div containing correlation visualizations
        """
        # Check if we have merged data
        if not hasattr(self.analysis_results, 'merged_data') or self.analysis_results.merged_data is None:
            return html.Div([
                html.H2("Correlation Analysis"),
                html.P("No data available. Please run the analysis pipeline first.",
                      style={'color': 'red'})
            ])
        
        df = self.analysis_results.merged_data
        
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Create correlation heatmap
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        heatmap_fig.update_layout(
            title={
                'text': 'Correlation Heatmap',
                'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Variables',
            yaxis_title='Variables',
            template='plotly_white',
            height=650,
            width=850,
            margin=dict(l=150, r=40, t=80, b=150),
            plot_bgcolor='#fafbfc',
            paper_bgcolor='#ffffff'
        )
        
        # Create scatter plots for key relationships
        scatter_plots = []
        
        # Interest rate vs credit relationships
        rate_cols = [col for col in df.columns if 'rate' in col.lower() or 'funds' in col.lower()]
        credit_cols = [col for col in df.columns if 'credit' in col.lower()]
        
        # Helper function to create meaningful titles and descriptions
        def get_chart_info(rate_col, credit_col):
            """Generate meaningful title and description for scatter plot."""
            # Create readable names
            rate_name = rate_col.replace('_', ' ').title()
            credit_name = credit_col.replace('_', ' ').title()
            
            # Generate title
            title = f"Impact of {rate_name} on {credit_name}"
            
            # Generate description based on relationship
            description = f"Each point represents a time period. Look for patterns: downward slopes suggest higher rates reduce credit, upward slopes suggest the opposite."
            
            return title, description
        
        for rate_col in rate_cols[:2]:  # Limit to first 2 rate columns
            for credit_col in credit_cols[:2]:  # Limit to first 2 credit columns
                if rate_col in df.columns and credit_col in df.columns:
                    title, description = get_chart_info(rate_col, credit_col)
                    
                    scatter_fig = go.Figure()
                    scatter_fig.add_trace(go.Scatter(
                        x=df[rate_col],
                        y=df[credit_col],
                        mode='markers',
                        marker=dict(size=8, color='#3498db', opacity=0.6),
                        name=f'{rate_col} vs {credit_col}'
                    ))
                    
                    scatter_fig.update_layout(
                        title={
                            'text': title,
                            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial, sans-serif'}
                        },
                        xaxis_title=rate_col.replace('_', ' ').title(),
                        yaxis_title=credit_col.replace('_', ' ').title(),
                        template='plotly_white',
                        height=400,
                        margin=dict(l=60, r=40, t=80, b=60)
                    )
                    
                    scatter_plots.append({
                        'graph': dcc.Graph(
                            figure=scatter_fig,
                            config={'displayModeBar': True, 'displaylogo': False}
                        ),
                        'description': description
                    })
        
        # Build panel with professional styling
        panel_children = [
            # Section Header
            html.Div([
                html.H2("🔗 Correlation Analysis", 
                       style={
                           'color': '#2c3e50',
                           'marginBottom': '10px',
                           'fontSize': '1.8rem',
                           'fontWeight': '600'
                       }),
                html.P("Explore relationships between interest rates and consumer credit variables.",
                      style={'color': '#7f8c8d', 'fontSize': '1rem', 'marginBottom': '25px'})
            ]),
            
            # Heatmap Card
            html.Div([
                dcc.Graph(
                    id='correlation-heatmap',
                    figure=heatmap_fig,
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '25px',
                'marginBottom': '30px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            }),
            
            # Scatter Plots Section
            html.H3("Detailed Relationships", 
                   style={
                       'color': '#34495e',
                       'marginTop': '20px',
                       'marginBottom': '20px',
                       'fontSize': '1.4rem',
                       'fontWeight': '600'
                   }),
        ]
        
        # Add scatter plots in cards with descriptions
        for plot_info in scatter_plots:
            panel_children.append(
                html.Div([
                    html.P(plot_info['description'], 
                          style={
                              'color': '#7f8c8d',
                              'fontSize': '0.95rem',
                              'marginBottom': '15px',
                              'fontStyle': 'italic'
                          }),
                    plot_info['graph']
                ], style={
                    'backgroundColor': '#ffffff',
                    'borderRadius': '12px',
                    'padding': '25px',
                    'marginBottom': '20px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                    'border': '1px solid #e9ecef'
                })
            )
        
        return html.Div(panel_children, style={'marginBottom': 40})
    
    def create_regression_panel(self) -> html.Div:
        """
        Create regression results display panel.
        
        Returns:
            html.Div containing regression model summaries
        """
        # Check if we have regression results
        if not hasattr(self.analysis_results, 'regression_results') or self.analysis_results.regression_results is None:
            return html.Div([
                html.H2("Regression Analysis"),
                html.P("No regression results available. Please run the regression analysis first.",
                      style={'color': 'red'})
            ])
        
        results = self.analysis_results.regression_results
        
        # Create coefficients table
        coef_rows = []
        for var, coef in results.coefficients.items():
            p_val = results.p_values.get(var, None)
            sig = ""
            if p_val is not None:
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
            
            coef_rows.append(html.Tr([
                html.Td(var, style={
                    'padding': '12px 16px',
                    'borderBottom': '1px solid #e9ecef',
                    'fontWeight': '500',
                    'color': '#34495e'
                }),
                html.Td(f"{coef:.6f}", style={
                    'padding': '12px 16px',
                    'borderBottom': '1px solid #e9ecef',
                    'textAlign': 'right',
                    'fontFamily': 'monospace',
                    'color': '#2c3e50'
                }),
                html.Td(f"{p_val:.6f}" if p_val is not None else "N/A", style={
                    'padding': '12px 16px',
                    'borderBottom': '1px solid #e9ecef',
                    'textAlign': 'right',
                    'fontFamily': 'monospace',
                    'color': '#2c3e50'
                }),
                html.Td(sig, style={
                    'padding': '12px 16px',
                    'borderBottom': '1px solid #e9ecef',
                    'textAlign': 'center',
                    'fontWeight': 'bold',
                    'color': '#e74c3c' if sig else '#95a5a6',
                    'fontSize': '1.1rem'
                })
            ]))
        
        coef_table = html.Table([
            html.Thead(html.Tr([
                html.Th("Variable", style={
                    'padding': '12px 16px',
                    'borderBottom': '2px solid #667eea',
                    'textAlign': 'left',
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'color': '#2c3e50'
                }),
                html.Th("Coefficient", style={
                    'padding': '12px 16px',
                    'borderBottom': '2px solid #667eea',
                    'textAlign': 'right',
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'color': '#2c3e50'
                }),
                html.Th("P-Value", style={
                    'padding': '12px 16px',
                    'borderBottom': '2px solid #667eea',
                    'textAlign': 'right',
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'color': '#2c3e50'
                }),
                html.Th("Sig.", style={
                    'padding': '12px 16px',
                    'borderBottom': '2px solid #667eea',
                    'textAlign': 'center',
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': '600',
                    'color': '#2c3e50'
                })
            ])),
            html.Tbody(coef_rows)
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse',
            'marginTop': '20px',
            'fontSize': '0.95rem'
        })
        
        # Model statistics
        stats_div = html.Div([
            html.H3("Model Statistics", style={'color': '#34495e', 'marginTop': 30}),
            html.Div([
                html.P([html.Strong("R-squared: "), f"{results.r_squared:.4f}"]),
                html.P([html.Strong("Adjusted R-squared: "), f"{results.adjusted_r_squared:.4f}"])
            ])
        ])
        
        # Diagnostic tests
        diagnostics_div = html.Div([
            html.H3("Diagnostic Tests", style={'color': '#34495e', 'marginTop': 30}),
            html.Div([
                html.P([html.Strong("Durbin-Watson: "), f"{results.diagnostics.durbin_watson:.4f}"]),
                html.P([html.Strong("Breusch-Pagan p-value: "), f"{results.diagnostics.breusch_pagan_p:.4f}"]),
                html.P([html.Strong("Jarque-Bera p-value: "), f"{results.diagnostics.jarque_bera_p:.4f}"])
            ])
        ])
        
        # Significance legend
        legend_div = html.Div([
            html.P([
                html.Small("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
            ], style={'color': '#7f8c8d', 'marginTop': 20})
        ])
        
        return html.Div([
            # Section Header
            html.Div([
                html.H2("📊 Regression Analysis", 
                       style={
                           'color': '#2c3e50',
                           'marginBottom': '10px',
                           'fontSize': '1.8rem',
                           'fontWeight': '600'
                       }),
                html.P("Model coefficients, statistics, and diagnostic tests.",
                      style={'color': '#7f8c8d', 'fontSize': '1rem', 'marginBottom': '25px'})
            ]),
            
            # Coefficients Card
            html.Div([
                html.H3("Model Coefficients", 
                       style={'color': '#34495e', 'marginBottom': '20px', 'fontSize': '1.3rem', 'fontWeight': '600'}),
                coef_table,
                legend_div
            ], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '30px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            }),
            
            # Statistics and Diagnostics Card
            html.Div([
                html.Div([
                    html.Div([stats_div], style={'flex': '1', 'minWidth': '250px'}),
                    html.Div([diagnostics_div], style={'flex': '1', 'minWidth': '250px'})
                ], style={'display': 'flex', 'gap': '40px', 'flexWrap': 'wrap'})
            ], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '30px',
                'marginBottom': '40px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            })
        ])
    
    def create_forecast_panel(self) -> html.Div:
        """
        Create forecast visualization panel.
        
        Returns:
            html.Div containing forecast visualizations with adjustable confidence intervals
        """
        # Check if we have forecast results
        if not hasattr(self.analysis_results, 'forecast_results') or self.analysis_results.forecast_results is None:
            return html.Div([
                html.H2("Forecast Analysis"),
                html.P("No forecast results available. Please run the forecast analysis first.",
                      style={'color': 'red'})
            ])
        
        forecast = self.analysis_results.forecast_results
        
        # Create forecast plot
        forecast_fig = go.Figure()
        
        # Add historical data if available
        if hasattr(self.analysis_results, 'merged_data') and self.analysis_results.merged_data is not None:
            df = self.analysis_results.merged_data
            # Assume we're forecasting total credit
            if 'total_credit' in df.columns:
                forecast_fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['total_credit'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#2c3e50', width=2)
                ))
        
        # Add forecast
        forecast_fig.add_trace(go.Scatter(
            x=forecast.dates,
            y=forecast.predicted_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#e74c3c', width=2, dash='dash')
        ))
        
        # Add confidence intervals
        forecast_fig.add_trace(go.Scatter(
            x=forecast.dates,
            y=forecast.upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        forecast_fig.add_trace(go.Scatter(
            x=forecast.dates,
            y=forecast.lower_bound,
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(231, 76, 60, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
        
        # Add confidence band legend entry
        forecast_fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=f'Confidence Band ({forecast.confidence_level*100:.0f}%)',
            marker=dict(size=10, color='rgba(231, 76, 60, 0.2)')
        ))
        
        forecast_fig.update_layout(
            title='Credit Forecast with Confidence Intervals',
            xaxis_title='Date',
            yaxis_title='Credit (Billions $)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        # Accuracy metrics if available
        metrics_div = html.Div()
        if hasattr(forecast, 'accuracy_metrics') and forecast.accuracy_metrics is not None:
            metrics = forecast.accuracy_metrics
            metrics_div = html.Div([
                html.H3("Forecast Accuracy Metrics", style={'color': '#34495e', 'marginTop': 30}),
                html.Div([
                    html.P([html.Strong("Mean Absolute Error (MAE): "), f"{metrics.mae:.4f}"]),
                    html.P([html.Strong("Root Mean Squared Error (RMSE): "), f"{metrics.rmse:.4f}"]),
                    html.P([html.Strong("Mean Absolute Percentage Error (MAPE): "), f"{metrics.mape:.4f}%"])
                ])
            ])
        
        return html.Div([
            # Section Header
            html.Div([
                html.H2("🔮 Forecast Analysis", 
                       style={
                           'color': '#2c3e50',
                           'marginBottom': '10px',
                           'fontSize': '1.8rem',
                           'fontWeight': '600'
                       }),
                html.P("Predicted credit trends with confidence intervals.",
                      style={'color': '#7f8c8d', 'fontSize': '1rem', 'marginBottom': '25px'})
            ]),
            
            # Forecast Chart Card
            html.Div([
                dcc.Graph(
                    id='forecast-chart',
                    figure=forecast_fig,
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '25px',
                'marginBottom': '20px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            }),
            
            # Metrics Card
            html.Div([metrics_div], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '30px',
                'marginBottom': '40px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            }) if metrics_div.children else html.Div()
        ])
    
    def create_scenario_simulator(self) -> html.Div:
        """
        Create interactive scenario simulator with sliders.
        
        Returns:
            html.Div containing interactive controls for scenario simulation
        """
        # Check if we have forecast results
        if not hasattr(self.analysis_results, 'forecast_results') or self.analysis_results.forecast_results is None:
            return html.Div([
                html.H2("Scenario Simulator"),
                html.P("No forecast results available. Please run the forecast analysis first.",
                      style={'color': 'red'})
            ])
        
        return html.Div([
            # Section Header
            html.Div([
                html.H2("🎯 Scenario Simulator", 
                       style={
                           'color': '#2c3e50',
                           'marginBottom': '10px',
                           'fontSize': '1.8rem',
                           'fontWeight': '600'
                       }),
                html.P("Adjust interest rate assumptions to see how credit projections change.",
                      style={'color': '#7f8c8d', 'fontSize': '1rem', 'marginBottom': '25px'})
            ]),
            
            # Simulator Card
            html.Div([
                html.Label("Interest Rate Adjustment (%)", 
                          style={
                              'fontWeight': '600',
                              'marginBottom': '15px',
                              'display': 'block',
                              'color': '#2c3e50',
                              'fontSize': '1.1rem'
                          }),
                dcc.Slider(
                    id='rate-adjustment-slider',
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    value=0.0,
                    marks={
                        -2.0: {'label': '-2%', 'style': {'color': '#e74c3c', 'fontWeight': 'bold'}},
                        -1.0: {'label': '-1%', 'style': {'color': '#e67e22'}},
                        0.0: {'label': '0%', 'style': {'color': '#95a5a6', 'fontWeight': 'bold'}},
                        1.0: {'label': '+1%', 'style': {'color': '#3498db'}},
                        2.0: {'label': '+2%', 'style': {'color': '#2980b9', 'fontWeight': 'bold'}}
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                
                html.Div(id='rate-adjustment-display', 
                        style={
                            'textAlign': 'center',
                            'marginTop': '25px',
                            'fontSize': '1.3rem',
                            'color': '#2c3e50',
                            'fontWeight': '600',
                            'padding': '15px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '8px',
                            'border': '2px solid #e9ecef'
                        }),
                
                html.Div(style={'marginTop': '30px', 'marginBottom': '20px', 'borderTop': '2px solid #e9ecef'}),
                
                html.Div(id='scenario-forecast-chart')
            ], style={
                'backgroundColor': '#ffffff',
                'borderRadius': '12px',
                'padding': '30px',
                'marginBottom': '40px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.08)',
                'border': '1px solid #e9ecef'
            })
        ])
    
    def register_callbacks(self):
        """
        Register all interactive component callbacks.
        
        Connects user inputs to analysis updates for interactive components.
        """
        from dash.dependencies import Input, Output
        
        # Callback for scenario simulator slider
        @self.app.callback(
            Output('rate-adjustment-display', 'children'),
            Input('rate-adjustment-slider', 'value')
        )
        def update_rate_display(rate_adjustment):
            """Update the rate adjustment display text."""
            if rate_adjustment is None:
                rate_adjustment = 0.0
            
            sign = '+' if rate_adjustment >= 0 else ''
            return f"Interest Rate Adjustment: {sign}{rate_adjustment:.1f}%"
        
        # Callback for scenario forecast chart
        @self.app.callback(
            Output('scenario-forecast-chart', 'children'),
            Input('rate-adjustment-slider', 'value')
        )
        def update_scenario_forecast(rate_adjustment):
            """Update forecast chart based on rate adjustment."""
            if rate_adjustment is None:
                rate_adjustment = 0.0
            
            # Check if we have forecast results
            if not hasattr(self.analysis_results, 'forecast_results') or self.analysis_results.forecast_results is None:
                return html.P("No forecast data available.", style={'color': 'red'})
            
            forecast = self.analysis_results.forecast_results
            
            # Adjust forecast based on rate change
            # Simple adjustment: assume credit decreases when rates increase
            # This is a simplified model - in reality, you'd use the ForecastEngine.simulate_scenario()
            adjustment_factor = 1 - (rate_adjustment / 100) * 0.5  # 0.5% credit change per 1% rate change
            adjusted_values = forecast.predicted_values * adjustment_factor
            adjusted_lower = forecast.lower_bound * adjustment_factor
            adjusted_upper = forecast.upper_bound * adjustment_factor
            
            # Create adjusted forecast plot
            scenario_fig = go.Figure()
            
            # Add historical data if available
            if hasattr(self.analysis_results, 'merged_data') and self.analysis_results.merged_data is not None:
                df = self.analysis_results.merged_data
                if 'total_credit' in df.columns:
                    scenario_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['total_credit'],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='#2c3e50', width=2)
                    ))
            
            # Add adjusted forecast
            scenario_fig.add_trace(go.Scatter(
                x=forecast.dates,
                y=adjusted_values,
                mode='lines',
                name='Adjusted Forecast',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            # Add confidence intervals
            scenario_fig.add_trace(go.Scatter(
                x=forecast.dates,
                y=adjusted_upper,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            scenario_fig.add_trace(go.Scatter(
                x=forecast.dates,
                y=adjusted_lower,
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(231, 76, 60, 0.2)',
                fill='tonexty',
                showlegend=False
            ))
            
            scenario_fig.update_layout(
                title=f'Scenario Forecast (Rate Adjustment: {rate_adjustment:+.1f}%)',
                xaxis_title='Date',
                yaxis_title='Credit (Billions $)',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            return dcc.Graph(
                figure=scenario_fig,
                config={'displayModeBar': True, 'displaylogo': False}
            )
    
    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        """
        Start Dash server.
        
        Args:
            host: Host address to run the server on (default: "127.0.0.1")
            port: Port number to run the server on (default: 8050)
            debug: Whether to run in debug mode (default: False)
        """
        # Update layout with all panels
        self.app.layout = html.Div([
            html.H1("Interest Rate and Consumer Credit Analysis Dashboard",
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            # Add all panels
            self.create_time_series_panel(),
            html.Hr(),
            self.create_correlation_panel(),
            html.Hr(),
            self.create_regression_panel(),
            html.Hr(),
            self.create_forecast_panel(),
            html.Hr(),
            self.create_scenario_simulator()
        ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
        
        # Register callbacks
        self.register_callbacks()
        
        # Run the server
        print(f"Starting dashboard server at http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
