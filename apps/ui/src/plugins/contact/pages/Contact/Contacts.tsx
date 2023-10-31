import styled from 'styled-components'
import { useNavigate } from 'react-router-dom'

import { ButtonPrimary } from 'components/Button/Button'
import ComponentsWrapper from 'components/ComponentsWrapper/ComponentsWrapper'

import IconButton from '@l3-lib/ui-core/dist/IconButton'

import MenuButton from '@l3-lib/ui-core/dist/MenuButton'

import Tab from '@l3-lib/ui-core/dist/Tab'
import TabList from '@l3-lib/ui-core/dist/TabList'
import TabPanel from '@l3-lib/ui-core/dist/TabPanel'
import TabPanels from '@l3-lib/ui-core/dist/TabPanels'
import TabsContext from '@l3-lib/ui-core/dist/TabsContext'

import {
  StyledHeaderGroup,
  StyledSectionTitle,
  StyledSectionWrapper,
} from 'pages/Home/homeStyle.css'
import { useContacts } from './useContacts'

import {
  StyledDeleteIcon,
  StyledEditIcon,
} from 'pages/TeamOfAgents/TeamOfAgentsCard/TeamOfAgentsCard'

import { useMemo, useState } from 'react'
import Table from 'components/Table'
import Groups, { StyledTableButtons } from '../Group/Groups'

import Microphone from '@l3-lib/ui-core/dist/icons/Microphone'

import { StyledCloseIcon } from 'pages/Home/GetStarted/GetStartedContainer'

import { useAgents } from 'pages/Agents/useAgents'
import AgentChatCard from 'components/ChatCards/AgentChatCard'
import { useModal } from 'hooks'
import { useContactForm } from './ContactForm/useContactForm'
import { StyledTabListSpan, StyledTabListWrapper, StyledTabRootWrapper } from 'styles/tabStyles.css'
import { t } from 'i18next'

const Contacts = () => {
  const navigate = useNavigate()

  const { contacts, deleteContactHandler, handleCall, handleEndCall } = useContacts()
  const { groupOptions } = useContactForm()

  const { agentsData } = useAgents()

  const { openModal } = useModal()

  const gridData =
    contacts?.map((contact: any) => ({
      id: contact.id,
      name: contact.name,
      description: contact.description,
      email: contact.email,
      phone: contact.phone,
      group_id: contact.group_id,
    })) || []

  const columns = useMemo(
    () => [
      {
        Header: 'Name',
        accessor: 'name',
        width: 225,
      },
      {
        Header: 'Phone',
        accessor: 'phone',
        width: 200,
        Cell: ({ cell }: any) => {
          return (
            <StyledPhoneCell>
              <StyledPhoneText>{cell.value}</StyledPhoneText>
              <StyledTableButtons>
                <MenuButton component={StyledCallIcon} closeDialogOnContentClick={false} zIndex={1}>
                  <StyledMenuList>
                    {agentsData?.map((agentObj: any, index: number) => {
                      const { agent } = agentObj

                      const handleView = () => {
                        openModal({
                          name: 'agent-view-modal',
                          data: {
                            agent: agentObj,
                          },
                        })
                      }

                      return (
                        <AgentChatCard
                          key={index}
                          onClick={handleCall}
                          onViewClick={handleView}
                          picked={false}
                          agent={agent}
                        />
                      )
                    })}
                  </StyledMenuList>
                </MenuButton>

                <IconButton
                  onClick={handleEndCall}
                  icon={() => <StyledCloseIcon size={25} />}
                  size={IconButton.sizes.SMALL}
                  kind={IconButton.kinds.TERTIARY}
                  ariaLabel='Hung up'
                />
              </StyledTableButtons>
            </StyledPhoneCell>
          )
        },
      },
      {
        Header: 'Email',
        accessor: 'email',
        width: 200,
      },
      {
        Header: 'Group',
        accessor: 'group_id',
        width: 100,
        Cell: ({ cell }: any) => {
          return (
            <span>
              {groupOptions
                ?.filter((group: any) => group.value === cell.value)
                .map((group: any) => group.label)}
            </span>
          )
        },
      },
      {
        Header: 'Description',
        accessor: 'description',
        width: 250,
      },
      {
        Header: 'Actions',
        accessor: 'id',
        width: 100,
        Cell: ({ cell }: any) => {
          return (
            <StyledTableButtons>
              <IconButton
                onClick={() => deleteContactHandler(cell.value)}
                icon={() => <StyledDeleteIcon />}
                size={IconButton.sizes.SMALL}
                kind={IconButton.kinds.TERTIARY}
                // ariaLabel='Delete'
              />

              <IconButton
                onClick={() => navigate(`/contacts/${cell.value}/edit-contact`)}
                icon={() => <StyledEditIcon />}
                size={IconButton.sizes.SMALL}
                kind={IconButton.kinds.TERTIARY}
                // ariaLabel='Edit'
              />
            </StyledTableButtons>
          )
        },
      },
    ],
    [],
  )

  const [activeTab, setActiveTab] = useState(0)

  return (
    <StyledTabRootWrapper>
      <StyledTabListWrapper>
        <TabList>
          <Tab onClick={() => setActiveTab(0)}>
            <StyledTabListSpan>{`${t('contact')}s`}</StyledTabListSpan>
          </Tab>
          <Tab onClick={() => setActiveTab(1)}>
            <StyledTabListSpan>{`${t('group')}s`}</StyledTabListSpan>
          </Tab>
        </TabList>
      </StyledTabListWrapper>

      <TabsContext activeTabId={activeTab}>
        <TabPanels noAnimation>
          <TabPanel>
            <StyledSectionWrapper>
              <StyledHeaderGroup className='header_group'>
                <div>
                  <StyledSectionTitle>{`${t('contact')}s`}</StyledSectionTitle>
                </div>
                <div>
                  <ButtonPrimary
                    onClick={() => navigate('/contacts/create-contact')}
                    size={'small'}
                  >
                    {t('add-contact')}
                  </ButtonPrimary>
                </div>
              </StyledHeaderGroup>

              <ComponentsWrapper noPadding>
                <StyledTableWrapper>
                  <Table columns={columns} data={gridData} />
                </StyledTableWrapper>
              </ComponentsWrapper>
            </StyledSectionWrapper>
          </TabPanel>

          <TabPanel>
            <Groups />
          </TabPanel>
        </TabPanels>
      </TabsContext>
    </StyledTabRootWrapper>
  )
}

export default Contacts

const StyledPhoneCell = styled.div`
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  height: 100%;
`
const StyledCallIcon = styled(Microphone)`
  path {
    fill: ${({ theme }) => theme.body.iconColor};
  }
`
const StyledMenuList = styled.div`
  /* width: 100px;
  height: 100px; */
  padding: 10px;
  overflow: auto;

  max-height: 300px;

  background: ${({ theme }) => theme.body.backgroundColorSecondary};
  border: ${({ theme }) => theme.body.secondaryBorder};
  backdrop-filter: blur(100px);
  border-radius: 10px;

  display: flex;
  flex-direction: column;
  gap: 5px;
`
const StyledPhoneText = styled.span`
  max-width: 60%;
  overflow: hidden;
`
export const StyledTableWrapper = styled.div`
  width: 100%;
  height: 100%;
  padding: 0 15px;
  overflow: auto;
  max-height: calc(100vh - 325px);
`