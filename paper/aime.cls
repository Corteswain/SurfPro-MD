%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LaTeX class for AIME pre-prints and manuscripts
% AI Laboratory for Molecular Engineering (AIME)
% Chalmers University of Technology
% Last updated: April 3rd, 2026
%
% Version: 2.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aime}[2026/04/03 AIME Lab manuscript class v2.1]
\LoadClassWithOptions{article}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[english]{babel}
\RequirePackage{etoolbox}
\RequirePackage{enumitem}

% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{bm}

% Tuned booktabs spacing
\setlength{\heavyrulewidth}{0.10em}
\setlength{\lightrulewidth}{0.05em}
\setlength{\cmidrulewidth}{0.03em}
\setlength{\abovetopsep}{0pt}
\setlength{\belowbottomsep}{0pt}
\addtolength{\aboverulesep}{0.15ex}
\addtolength{\belowrulesep}{0.15ex}

% Math %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{definition}{Definition}

% Algorithms %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{algorithm}
\RequirePackage{algorithmic}

% Color palette — AIME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{xcolor}
\RequirePackage[most]{tcolorbox}
\RequirePackage{tikz}

% Primary palette
\definecolor{aimedeepteal}{HTML}{2D4F54}
\definecolor{aimeroyalpurple}{HTML}{7B539E}
\definecolor{aimelavender}{HTML}{B8A0D2}
\definecolor{aimeforestgreen}{HTML}{5A9448}
\definecolor{aimemutedplum}{HTML}{9E4A78}
\definecolor{aimedarkslate}{HTML}{3A3D4A}

% Derived colors
\definecolor{aimelightteal}{HTML}{E8F0F1}    % very light teal tint
\definecolor{aimelightlavender}{HTML}{EEEAF4} % very light lavender for title card
\definecolor{aimewhite}{RGB}{255, 255, 255}
\definecolor{aimeblack}{RGB}{0, 0, 0}
\definecolor{aimegray}{HTML}{6B6B6B}

% Hyperlinks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}
\hypersetup{
  colorlinks,
  linkcolor=aimemutedplum,
  citecolor=aimedeepteal,
  urlcolor=aimemutedplum
}

% ─────────────────────────────────────────────────────────────────────
% Section formatting — colored section numbers
% The number renders in Deep Teal; the title text in Dark Slate.
% ─────────────────────────────────────────────────────────────────────

\RequirePackage{titlesec}

\titleformat{\section}
  {\Large\sffamily\bfseries\color{aimedarkslate}}
  {\thesection}{0.6em}{}

\titleformat{\subsection}
  {\large\sffamily\bfseries\color{aimedarkslate}}
  {\thesubsection}{0.5em}{}

\titleformat{\subsubsection}
  {\normalsize\sffamily\bfseries\color{aimedarkslate}}
  {\thesubsubsection}{0.5em}{}

\titleformat*{\paragraph}{\itshape}

\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

% Captions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Figure 1" / "Table 1" label in Deep Teal, caption text in dark slate.

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{ }
\DeclareCaptionFormat{custom}{{\sffamily\color{aimedeepteal}\textbf{#1#2}}#3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}

% ─────────────────────────────────────────────────────────────────────
% Running header on pages 2+
% Thin teal rule with short title in small caps on the right.
% ─────────────────────────────────────────────────────────────────────

\RequirePackage{fancyhdr}
\RequirePackage{lastpage}

% Store the short title for the running header
\newcommand{\@aime@shorttitle}{}
\newcommand{\shorttitle}[1]{\renewcommand{\@aime@shorttitle}{#1}}

\fancypagestyle{aime@body}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \fancyhead[L]{\textcolor{aimedeepteal}{\rule{\textwidth}{0.4pt}}}%
  \fancyhead[R]{\vspace{0.15cm}\small\sffamily\color{aimegray}\@aime@shorttitle\hspace{0.4cm}\thepage}%
  \fancyfoot{}%
}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \fancyfoot[C]{\small\thepage}%
}

\pagestyle{aime@body}

% ─────────────────────────────────────────────────────────────────────
% Authors, affiliations, metadata
% ─────────────────────────────────────────────────────────────────────

\newcommand{\@aime@title}{}
\newcommand{\@aime@authors}{}
\newif\if@aime@firstauthor \@aime@firstauthortrue
\newcommand{\@aime@affiliations}{}
\newif\if@aime@firstaffil \@aime@firstaffiltrue
\newcommand{\@aime@contributions}{}
\newif\if@aime@firstcontrib \@aime@firstcontribtrue
\newcommand{\@aime@abstract}{}
\newcommand{\@aime@metaline}{}
\newif\if@aime@firstmeta \@aime@firstmetatrue

% Public commands
\renewcommand{\title}[1]{\renewcommand{\@aime@title}{#1}}

\newcommand{\@aime@addauthor}[2][]{%
  \if@aime@firstauthor
    \gdef\@aime@authors{{\sffamily\bfseries #2}$^{#1}$}%
    \global\@aime@firstauthorfalse
  \else
    \g@addto@macro\@aime@authors{, {\sffamily\bfseries #2}$^{#1}$}%
  \fi
}
\renewcommand{\author}[2][]{\@aime@addauthor[#1]{#2}}

\newcommand{\@aime@addaffil}[2][]{%
  \if@aime@firstaffil
    \gdef\@aime@affiliations{$^{#1}$#2}%
    \global\@aime@firstaffilfalse
  \else
    \g@addto@macro\@aime@affiliations{\par $^{#1}$#2}%
  \fi
}
\newcommand{\affiliation}[2][]{\@aime@addaffil[#1]{#2}}

\newcommand{\@aime@addcontrib}[2][]{%
  \if@aime@firstcontrib
    \gdef\@aime@contributions{{\small $^{#1}$#2}}%
    \global\@aime@firstcontribfalse
  \else
    \g@addto@macro\@aime@contributions{, {\small $^{#1}$#2}}%
  \fi
}
\newcommand{\contribution}[2][]{\@aime@addcontrib[#1]{#2}}

% Metadata stored as stacked lines
\newcommand{\@aime@addmeta}[2]{%
  \if@aime@firstmeta
    \gdef\@aime@metaline{{\small {\sffamily\bfseries #1:} #2}}%
    \global\@aime@firstmetafalse
  \else
    \g@addto@macro\@aime@metaline{\par{\small {\sffamily\bfseries #1:} #2}}%
  \fi
}

\renewcommand{\abstract}[1]{\renewcommand{\@aime@abstract}{#1}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\renewcommand{\date}[1]{\@aime@addmeta{Date}{#1}}
\newcommand{\correspondence}[1]{\@aime@addmeta{Correspondence}{#1}}
\newcommand{\codeavailability}[1]{\@aime@addmeta{Code}{#1}}
\newcommand{\dataavailability}[1]{\@aime@addmeta{Data}{#1}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}

% ─────────────────────────────────────────────────────────────────────
% Title block — tcolorbox card with rounded corners
% Soft lavender background, no frame, no logo by default.
% Contains title, authors, affiliations, abstract, and metadata.
% ─────────────────────────────────────────────────────────────────────

\makeatletter
\newcommand{\@aime@maketitle}{%
  \tcbset{
    enhanced,
    frame hidden,
    left=0.5cm,
    right=0.5cm,
    top=0.5cm,
    bottom=0.5cm,
    arc=10pt,
    colback=aimelightlavender,
    before skip=0pt,
    grow to left by=1.5pt,
    grow to right by=1.5pt,
  }%
  % Uncomment to add a logo (not for anonymous submissions):
  % \tcbset{overlay={\node[
  %   anchor=south east,
  %   at=(frame.south east),
  %   xshift=-0.5cm,
  %   yshift=0.5cm] {\includegraphics[width=1.5cm]{assets/logo-aime.png}};}}
  %
  \begin{tcolorbox}
    \setlength{\parindent}{0cm}
    \setlength{\parskip}{0.5cm}
    {%
      \setlength{\parskip}{0cm}
      \raggedright
      \nohyphens
      %
      % ── Title ──
      {%
        \setstretch{1.15}
        {\huge\sffamily\bfseries\color{aimedarkslate}\@aime@title\par}%
      }
      \vskip 0.2cm
      %
      % ── Authors ──
      \@aime@authors\par
      \vskip 0.2cm
      %
      % ── Affiliations ──
      {\small\@aime@affiliations\par}%
      %
      % ── Contributions ──
      \if@aime@firstcontrib\else
        \vskip 0.05cm
        \@aime@contributions\par
      \fi
    }%
    %
    % ── Abstract ──
    {\sffamily\bfseries\small\color{aimedeepteal}Abstract\hspace{0.5em}}%
    \@aime@abstract\par
    %
    % ── Metadata ──
    \vskip 0.5cm
    {%
      \setlength{\parskip}{0cm}
      \if@aime@firstmeta\vspace*{0.65cm}\else\@aime@metaline\ \\\par\fi
    }%
  \end{tcolorbox}
  \tcbset{reset}
  \FloatBarrier
}

\if@twocolumn%
\renewcommand{\maketitle}{
  \thispagestyle{plain}%
  \twocolumn[%
    \@aime@maketitle
    \vskip 0.38cm
  ]
}%
\else%
\renewcommand{\maketitle}{
  \thispagestyle{plain}%
  \@aime@maketitle
}%
\fi
\makeatother

% Natbib for references %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[numbers,sort&compress]{natbib}
\renewcommand{\bibfont}{\small}

% Acknowledgments environment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{ack}{%
  \section*{Acknowledgments}
}{}
